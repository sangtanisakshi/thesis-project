
import numpy as np
import pandas as pd
import glob
import torch
import time
import warnings
import argparse
import pickle
import os
import sys
sys.path.append(".")
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from model.evaluation import get_utility_metrics,stat_sim
from model.data_preparation import DataPrep
from sklearn.preprocessing import LabelEncoder
from thesisgan.model_evaluation import eval_model
from PIL import Image
import wandb
import optuna
import joblib
from optuna.samplers import TPESampler
from types import SimpleNamespace
os.environ['WANDB_OFFLINE'] = 'true'

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run", type=str, default="ctabgan_sweep_run")
    parser.add_argument("--desc", type=str, default="HPO Run for CTABGAN with optuna")
    parser.add_argument("--hpo", default=False)
    parser.add_argument("--test_ratio", type=float, default=0.00003)
    parser.add_argument("--categorical_columns", nargs='+', default=['proto', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'label', 'attack_type'])
    parser.add_argument("--log_columns", nargs='+', default=[])
    parser.add_argument("--mixed_columns", nargs='+', default={})
    parser.add_argument("--general_columns", nargs='+', default=[])
    parser.add_argument("--non_categorical_columns", nargs='+', default= ['packets','src_pt','dst_pt','duration','bytes'])
    parser.add_argument("--integer_columns", nargs='+', default=[])
    parser.add_argument("--problem_type", type=dict, default={"Classification": 'attack_type'})
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--save", default=True)
    parser.add_argument("--random_dim", type=int, default=100)
    parser.add_argument("--class_dim", type=str, default=(256, 256, 256, 256))
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("-bs", "--batch_size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("-ns", "--num_samples", type=int, default=None)
    parser.add_argument("-lr", "--lr", type=float, default=2e-4)
    parser.add_argument("-lr_betas", "--lr_betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("-eps", "--eps", type=float, default=1e-3)
    parser.add_argument("--lambda_", type=float, default=10)
    parser.add_argument("--ip_path", type=str, default="thesisgan/input/new_train_data.csv")
    parser.add_argument("--op_path", type=str, default="thesisgan/output/")
    parser.add_argument("--test_data", type=str, default="thesisgan/input/new_test_data.csv")
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()
    return args

def get_hpo_parameters(trial):
    
    #defining the hyperparameters that need tuning
    random_dim = trial.suggest_categorical('random_dim',[50, 100, 200])
    class_dim = trial.suggest_categorical('class_dim',[(256,256,256,256), (512,512,512,512)])
    num_channels = trial.suggest_categorical('num_channels',[32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay",1e-6,1e-3,log=True)
    batch_size = trial.suggest_categorical('batch_size',[500, 1000])
    lr = trial.suggest_float("lr",1e-4,1e-3,log=True)
    lr_betas = trial.suggest_categorical('lr_betas',[(0.5, 0.9), (0.5, 0.95), (0.9, 0.999)])
    eps = trial.suggest_categorical('eps',[1e-3, 1e-4, 1e-5])
    lambda_ = trial.suggest_categorical('lambda_',[10, 20, 30])
    hpo_params = dict({"random_dim": random_dim, "class_dim": class_dim, "num_channels": num_channels, "weight_decay": weight_decay, "batch_size": batch_size, "lr": lr, "lr_betas": lr_betas, "eps": eps, "lambda_": lambda_})
    print("Hyperparameters for current trial: ",hpo_params)
    return hpo_params

def wrapper(train_data, rest_args):
    def hpo(trial):
        hpo_params = get_hpo_parameters(trial)
        config = {}
        config.update(rest_args)
        config.update(hpo_params)
        print(config)
        wandb.init(project="masterthesis",
                    config=config,
                    group="CTABGAN_SWEEP_OPTUNA", 
                    notes=rest_args["desc"],
                    mode="offline")
        synthesizer = CTABGANSynthesizer(config)
        start_time = time.time()
        data_prep = DataPrep(train_data,rest_args["categorical_columns"],rest_args["log_columns"],
                                rest_args["mixed_columns"],["general_columns"],
                                rest_args["non_categorical_columns"],rest_args["integer_columns"],
                            rest_args["problem_type"],rest_args["test_ratio"])
        print("Data preparation complete. Time taken: ",time.time()-start_time)
        print("Starting training...")
        train_time = time.time()
        gan_loss = synthesizer.fit(train_data=data_prep.df, categorical = data_prep.column_types["categorical"], mixed = data_prep.column_types["mixed"],
        general = data_prep.column_types["general"], non_categorical = data_prep.column_types["non_categorical"], type=rest_args["problem_type"])
        end_time = time.time()
        print('Finished training in',end_time-train_time," seconds.")
        wandb.log({"training_time": end_time-train_time})
        wandb.log({"WGAN-GP_experiment": gan_loss, "trial": trial.number})
        trial = str(trial.number)
        op_path = (rest_args['op_path'] + rest_args['wandb_run'] + "/" + trial + "/")
        test_data, syn_data = sample_data(synthesizer, rest_args, op_path, data_prep)
        eval(train_data, test_data, syn_data, op_path)
        wandb.finish()
        return gan_loss
    return hpo


def sample_data(model, args, op_path, data_prep):
    
    os.makedirs(op_path, exist_ok=True)
    print("Saving?: ",args["save"])
    if args["save"]:
        pickle.dump(model, open(str(op_path+"pklmodel.pkl"), "wb"))

    # Load test data
    test_data = pd.read_csv(args["test_data"])
    if test_data.columns[0] == "Unnamed: 0":
        test_data = test_data.drop(columns="Unnamed: 0")
    print("Test data loaded")
    
    n = test_data.shape[0] if args["num_samples"] is None else args["num_samples"]
    
    sample_start = time.time()
    
    sample = model.sample(n) 
    sample_df = data_prep.inverse_prep(sample)
    
    sample_end = time.time() - sample_start
    print(f"Sampling time: {sample_end} seconds")
    wandb.log({"sampling_time": sample_end})   
    print("Data sampling complete. Saving synthetic data...")
    
    sample_df.to_csv(str(op_path+"syn.csv"), index=False)
    print("Synthetic data saved. Check the output folder.")
    
    return test_data, sample_df

def eval(train_data, test_data, sample_data, op_path):
    eval_metadata = {
        "columns" : 
        {
        "duration": {"sdtype": "numerical", "compute_representation": "Float" },
        "proto": {"sdtype": "categorical"},
        "src_pt": {"sdtype": "numerical", "compute_representation": "Float"},
        "dst_pt": {"sdtype": "numerical", "compute_representation": "Float"},
        "packets": {"sdtype": "numerical", "compute_representation": "Float"},
        "bytes": {"sdtype": "numerical", "compute_representation": "Float"},
        "tcp_ack": {"sdtype": "categorical"},
        "tcp_psh": {"sdtype": "categorical"},
        "tcp_rst": {"sdtype": "categorical"},
        "tcp_syn": {"sdtype": "categorical"},
        "tcp_fin":{"sdtype": "categorical"},
        "tos": {"sdtype": "categorical"},
        "label": {"sdtype": "categorical"},
        "attack_type": {"sdtype": "categorical"}
        }
        }

    #remove columns with only one unique value
    for col in test_data.columns:
        if len(test_data[col].unique()) == 1:
            print(f"Removing column {col} as it has only one unique value")
            test_data.drop(columns=[col], inplace=True)
            sample_data.drop(columns=[col], inplace=True)
            
    le_dict = {"attack_type": "le_attack_type", "label": "le_label", "proto": "le_proto", "tos": "le_tos"}
    for c in le_dict.keys():
        le_dict[c] = LabelEncoder()
        test_data[c] = le_dict[c].fit_transform(test_data[c])
        sample_data[c] = le_dict[c].fit_transform(sample_data[c])
        train_data[c] = le_dict[c].fit_transform(train_data[c])
        
    cat_cols = ['proto', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'label', 'attack_type']
    for col in cat_cols:
        test_data[col] = test_data[col].astype(str)
        sample_data[col] = sample_data[col].astype(str)
    
    #Data Evaluation
    scores = eval_model(test_data, sample_data, eval_metadata, op_path)

    # #Save the evaluation results to wandb
    imgs = [np.asarray(Image.open(f)) for f in sorted(glob.glob(op_path + "*.png"))]
    wandb.log({"Evaluation": [wandb.Image(img) for img in imgs]})
    wandb.log(scores)
    
    for col in cat_cols:
        test_data[col] = test_data[col].astype("int64")
        sample_data[col] = sample_data[col].astype("int64")
        train_data[col] = train_data[col].astype("int64")
        
    #if the synthetic data does not have the same attack types as the test data, we need to fix it
    for at in test_data["attack_type"].unique():
        if at not in sample_data["attack_type"].unique():
            #add a row with the attack type
            sample_data = pd.concat([sample_data,train_data[train_data["attack_type"] == at].sample(3)], ignore_index=True)
    
    #if the synthetic data has only one unique value for the attack type, add a row with the attack type
    sample_data_value_counts = sample_data["attack_type"].value_counts()
    for i in range(len(sample_data_value_counts)):
        if sample_data_value_counts[i] == 1:
            at = sample_data_value_counts.index[i]
            sample_data = pd.concat([sample_data,train_data[train_data["attack_type"] == at].sample(3)], ignore_index=True)

    #Model Classification
    model_dict =  {"Classification":["xgb","lr","dt","rf","mlp"]}
    result_df, cr = get_utility_metrics(train_data,test_data,sample_data,"MinMax", model_dict)

    stat_res_avg = []
    stat_res = stat_sim(test_data, sample_data, cat_cols)
    stat_res_avg.append(stat_res)

    stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
    stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
    
    wandb.log({'Stat Results' : wandb.Table(dataframe=stat_results), 
               'Classification Results': wandb.Table(dataframe=result_df),
               'Classification Report': wandb.Table(dataframe=cr)})
    print("Evaluation complete. Check the output folder for the plots and evaluation results.")
    
if __name__ == "__main__":

    args = argument_parser()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    warnings.filterwarnings("ignore")
    
    seed_everything(args.seed)
    
    if args.hpo:
        args.ip_path = "thesisgan/input/new_hpo_data.csv"
        rest_args = {
        'ip_path': args.ip_path,
        'wandb_run': args.wandb_run,
        'desc': args.desc,
        'op_path': args.op_path,
        'test_data': args.test_data,
        'seed': args.seed,
        'epochs': args.epochs,
        'save': args.save,
        'num_samples': args.num_samples,
        'hpo': args.hpo,
        'categorical_columns': args.categorical_columns,
        'log_columns': args.log_columns,
        'mixed_columns': args.mixed_columns,
        'general_columns': args.general_columns,
        'non_categorical_columns': args.non_categorical_columns,
        'integer_columns': args.integer_columns,
        'problem_type': args.problem_type,
        'test_ratio': args.test_ratio,
        'n_trials': args.n_trials,
        }
        train_data = pd.read_csv(args.ip_path)
        le_dict = {"attack_type": "le_attack_type", "label": "le_label", "proto": "le_proto", "tos": "le_tos"}
        for c in le_dict.keys():
            le_dict[c] = LabelEncoder()
            train_data[c] = le_dict[c].fit_transform(train_data[c])
            train_data[c] = train_data[c].astype("int64")
        sampler = TPESampler(seed=123)  # Make the sampler behave in a deterministic way and get reproducable results
        study = optuna.create_study(direction="minimize",sampler=sampler)
        study.optimize(wrapper(train_data, rest_args), n_trials=rest_args["n_trials"])
        joblib.dump(study,("thesisgan/hpo_results/hyperparameter_optimization/trials_data/study.pkl"))
        study_data = pd.DataFrame(study.trials_dataframe())
        data_csv = study_data.to_csv("thesisgan/hpo_results/hyperparameter_optimization/trials_data/study.csv")
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial params:")
        for key, value in study.best_params.items():
            print(" {}: {}".format(key, value))
    
        #get best trial hyperparameters and train the model with that
        best_params = SimpleNamespace(**study.best_params)
    else:
        train_data = pd.read_csv(args.ip_path)
        print("Training non-hpo model")
        #convert args to a dictionary
        config = args.__dict__
        wandb.init(project="masterthesis",
                    config=config,
                    group="CTABGAN_SWEEP_OPTUNA", 
                    notes=config["desc"],
                    mode="offline")
        synthesizer = CTABGANSynthesizer(config)
        start_time = time.time()
        data_prep = DataPrep(train_data,config["categorical_columns"],config["log_columns"],
                                config["mixed_columns"],["general_columns"],
                                config["non_categorical_columns"],config["integer_columns"],
                            config["problem_type"],config["test_ratio"])
        print("Data preparation complete. Time taken: ",time.time()-start_time)
        print("Starting training...")
        train_time = time.time()
        gan_loss = synthesizer.fit(train_data=data_prep.df, categorical = data_prep.column_types["categorical"], mixed = data_prep.column_types["mixed"],
        general = data_prep.column_types["general"], non_categorical = data_prep.column_types["non_categorical"], type=config["problem_type"])
        end_time = time.time()
        print('Finished training in',end_time-train_time," seconds.")
        wandb.log({"training_time": end_time-train_time})
        wandb.log({"WGAN-GP_experiment": gan_loss})
        op_path = (config['op_path'] + config['wandb_run'] + "/")
        test_data, syn_data = sample_data(synthesizer, config, op_path, data_prep)
        eval(train_data, test_data, syn_data, op_path)
        wandb.finish()
       
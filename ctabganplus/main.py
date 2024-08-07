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
    parser.add_argument("--wandb_run", type=str, default="ctabgan_binary_1")
    parser.add_argument("--desc", type=str, default="Binary classification")
    parser.add_argument("--hpo", action=argparse.BooleanOptionalAction)
    parser.add_argument("--test_ratio", type=float, default=0.00003)
    parser.add_argument("--categorical_columns", nargs='+', default=['proto', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'label', 'attack_type'])
    parser.add_argument("--log_columns", nargs='+', default=[])
    parser.add_argument("--mixed_columns", nargs='+', default={})
    parser.add_argument("--general_columns", nargs='+', default=[])
    parser.add_argument("--non_categorical_columns", nargs='+', default= ['packets','src_pt','dst_pt','duration','bytes'])
    parser.add_argument("--integer_columns", nargs='+', default=[])
    parser.add_argument("--problem_type", type=dict, default={"Classification": 'label'})
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    parser.add_argument("--random_dim", type=int, default=200)
    parser.add_argument("--class_dim", type=str, default=(512,512,512,512))
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.00000215862158190158)
    parser.add_argument("-bs", "--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("-ns", "--num_samples", type=int, default=None)
    parser.add_argument("-lr", "--lr", type=float, default=0.0004969483674705974)
    parser.add_argument("-lr_betas", "--lr_betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("-eps", "--eps", type=float, default=0.00001)
    parser.add_argument("--lambda_", type=float, default=30)
    parser.add_argument("--ip_path", type=str, default="thesisgan/input/new_train_data.csv")
    parser.add_argument("--op_path", type=str, default="thesisgan/output/")
    parser.add_argument("--test_data", type=str, default="thesisgan/input/new_hpo_data.csv")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--cv", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

def get_hpo_parameters(trial):
    
    #defining the hyperparameters that need tuning
    random_dim = trial.suggest_categorical('random_dim',[50, 100, 200])
    class_dim = trial.suggest_categorical('class_dim',[(256,256,256,256), (512,512,512,512)])
    num_channels = trial.suggest_categorical('num_channels',[32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay",1e-6,1e-3,log=True)
    lr = trial.suggest_float("lr",1e-4,1e-3,log=True)
    lr_betas = trial.suggest_categorical('lr_betas',[(0.5, 0.9), (0.5, 0.95), (0.9, 0.999)])
    eps = trial.suggest_categorical('eps',[1e-3, 1e-4, 1e-5])
    lambda_ = trial.suggest_categorical('lambda_',[10, 20, 30])
    hpo_params = dict({"random_dim": random_dim, "class_dim": class_dim, "num_channels": num_channels, "weight_decay": weight_decay, "lr": lr, "lr_betas": lr_betas, "eps": eps, "lambda_": lambda_})
    print("Hyperparameters for current trial: ",hpo_params)
    return hpo_params

def wrapper(train_data, rest_args, test_data):
    def hpo(trial):
        hpo_params = get_hpo_parameters(trial)
        config = {}
        config.update(rest_args)
        config.update(hpo_params)
        print(config)
        wandb.init(project="masterthesis",
                    config=config,
                    group="CTABGAN_HPO_NEW", 
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
        syn_data = sample_data(synthesizer, rest_args, op_path, train_data, data_prep)
        eval(train_data, test_data, syn_data, op_path)
        wandb.finish()
        return gan_loss
    return hpo


def sample_data(model, args, op_path, train_data, data_prep, cv=False):
    
    os.makedirs(op_path, exist_ok=True)
    print("Saving?: ",args["save"])
    if args["save"]:
        pickle.dump(model, open(str(op_path+"pklmodel.pkl"), "wb"))

    n = train_data.shape[0] if args["num_samples"] is None else args["num_samples"]
    
    sample_start = time.time()
    
    sample = model.sample(n) 
    sample_df = data_prep.inverse_prep(sample)
    
    sample_end = time.time() - sample_start
    print(f"Sampling time: {sample_end} seconds")
    wandb.log({"sampling_time": sample_end})   
    print("Data sampling complete. Saving synthetic data...")
    
    sample_df.to_csv(str(op_path+"syn.csv"), index=False)
    print("Synthetic data saved. Check the output folder.")
    
    return sample_df

def eval(train_data, test_data, sample_data, op_path, cv=False, binary=False):
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
        "attack_type": {"sdtype": "categorical"},
        "label": {"sdtype": "categorical"},
        }
        }
    
    cat_cols = ['proto', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'label', 'attack_type']
    for col in cat_cols:
        test_data[col] = test_data[col].astype(str)
        sample_data[col] = sample_data[col].astype(str)
        train_data[col] = train_data[col].astype(str)
    
    #Data Evaluation
    scores = eval_model(test_data, sample_data, eval_metadata, op_path)

    # #Save the evaluation results to wandb
    imgs = [np.asarray(Image.open(f)) for f in sorted(glob.glob(op_path + "*.png"))]
    wandb.log({"Evaluation": [wandb.Image(img) for img in imgs]})
    wandb.log(scores)

    for col in cat_cols:
        test_data[col] = test_data[col].astype("float64")
        test_data[col] = test_data[col].astype("int64")
        sample_data[col] = sample_data[col].astype("float64")
        sample_data[col] = sample_data[col].astype("int64")
        train_data[col] = train_data[col].astype("float64")
        train_data[col] = train_data[col].astype("int64")
        
    if binary==False or cv==False:
        #if the synthetic data has only one unique value for the attack type, add a row with the attack type
        sample_data_value_counts = sample_data["attack_type"].value_counts()
        for i in range(len(sample_data_value_counts)):
            if sample_data_value_counts[i] == 1:
                at = sample_data_value_counts.index[i]
                sample_data = pd.concat([sample_data,train_data[train_data["attack_type"] == at].sample(3)], ignore_index=True)

        for at in test_data["attack_type"].unique():
            if at not in sample_data["attack_type"].unique():
                #add a row with the attack type
                sample_data = pd.concat([sample_data,train_data[train_data["attack_type"] == at].sample(3)], ignore_index=True)

    #Model Classification
    model_dict =  {"Classification":["xgb","lr","dt","rf","mlp"]}
    result_df, cr = get_utility_metrics(train_data,test_data,sample_data,"MinMax", model_dict, cv=cv, binary=binary)

    if cv==False:
        stat_res_avg = []
        stat_res = stat_sim(test_data, sample_data, cat_cols)
        stat_res_avg.append(stat_res)

        stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
        stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
    
        wandb.log({'Stat Results' : wandb.Table(dataframe=stat_results), 
               'Classification Results': wandb.Table(dataframe=result_df),
               'Classification Report': wandb.Table(dataframe=cr)})
    else:
        
        wandb.log({'Classification Results': wandb.Table(dataframe=result_df),
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
        'batch_size': args.batch_size
        }
    
    else:
        config = args.__dict__
        print("Config: ",config)

    if args.hpo:
        train_data = pd.read_csv(args.ip_path)
        test_data = pd.read_csv(args.test_data)
        attack_type_le = {"benign": 0, "bruteForce": 1, "portScan": 2, "pingScan": 3, "dos": 4}
        proto_le = {"TCP": 0, "UDP": 1, "ICMP": 2, "IGMP": 3}
        label_type_le = {"normal": 0, "attack": 1, "attacker": 1, "victim": 1}
        tos_le = {0 : 0, 32 : 1, 192 : 2, 16 : 3}
        #based on the unique values in the dataset, we will create a dictionary to map the values to integers
        datasets = [train_data, test_data]
        for dataset in datasets:
            dataset["attack_type"] = dataset["attack_type"].map(attack_type_le)
            dataset["proto"] = dataset["proto"].map(proto_le)
            dataset["tos"] = dataset["tos"].map(tos_le)
            dataset["label"] = dataset["label"].map(label_type_le)
            
        sampler = TPESampler(seed=123)  # Make the sampler behave in a deterministic way and get reproducable results
        study = optuna.create_study(direction="minimize",sampler=sampler)
        study.optimize(wrapper(train_data, rest_args, test_data), n_trials=rest_args["n_trials"])
        joblib.dump(study,("thesisgan/hpo_results/trials_data/ctabgan_study_2.pkl"))
        study_data = pd.DataFrame(study.trials_dataframe())
        data_csv = study_data.to_csv("thesisgan/hpo_results/trials_data/ctabgan_study_2.csv")
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial params:")
        for key, value in study.best_params.items():
            print(" {}: {}".format(key, value))
        # get best trial hyperparameters and train the model with that
        best_params = SimpleNamespace(**study.best_params)
        print("Best trial params:",best_params.__dict__)

    elif args.cv:
        print("Training with Cross Validation")
        # update args to set with the best hp of the model - trial 4 (new hpo)
        config.update({
            "problem_type": {"Classification": 'label'},
            "random_dim": 200,
            "epochs":1,
            "num_channels": 128,
            "eps": 0.00001,
            "lr": 0.0004969483674705974,
            "batch_size": 2000,
            "lambda_": 30,
            "seed": 23,
            "ip_path": "thesisgan/input/new_train_data.csv",
            "op_path": "thesisgan/output/ctab_cv_test/",
            "weight_decay": 0.00000215862158190158
        })
        train_data = pd.read_csv(config["ip_path"])
        #move the label at the end of the dataframe
        cols = list(train_data.columns)
        cols.remove("label")
        cols.append("label")
        train_data = train_data[cols]
        
        config.update({"wandb_run": f"CTABGAN_CV_test", "desc": "CTABGAN_CV_test run"})
        wandb.init(project="masterthesis", name=config["wandb_run"], config=config, notes=config["desc"],
                            group="CTABGAN-CV_test", mode="offline")
        print(f"Starting fold 1")
        test_df = train_data[train_data["attack_type"] == "bruteForce"]
        train_df = train_data[train_data["attack_type"] != "bruteForce"]
        
        # First we will convert the training data label to a binary classification algorithm
        # where normal traffic is 0 and attack traffic is 1
        attack_type_le = {"benign": 0, "bruteForce": 1, "portScan": 2, "pingScan": 3, "dos": 4}
        proto_le = {"TCP": 0, "UDP": 1, "ICMP": 2, "IGMP": 3}
        label_type_le = {"normal": 0, "attack": 1, "attacker": 1, "victim": 1}
        tos_le = {0 : 0, 32 : 1, 192 : 2, 16 : 3}
        
        #based on the unique values in the dataset, we will create a dictionary to map the values to integers
        datasets = [train_df, test_df]
        for dataset in datasets:
            dataset["attack_type"] = dataset["attack_type"].map(attack_type_le)
            dataset["proto"] = dataset["proto"].map(proto_le)
            dataset["tos"] = dataset["tos"].map(tos_le)
            dataset["label"] = dataset["label"].map(label_type_le)
        
        synthesizer = CTABGANSynthesizer(config)
        start_time = time.time()
        data_prep = DataPrep(train_df,config["categorical_columns"],config["log_columns"],
                            config["mixed_columns"],["general_columns"],
                            config["non_categorical_columns"],config["integer_columns"],
                        config["problem_type"],config["test_ratio"])
        print("Data preparation complete. Time taken: ",time.time()-start_time)
        print("Starting training...")
        train_time = time.time()
        gan_loss = synthesizer.fit(train_data=data_prep.df, categorical = data_prep.column_types["categorical"], mixed = data_prep.column_types["mixed"],
                                    general = data_prep.column_types["general"], 
                                    non_categorical = data_prep.column_types["non_categorical"], 
                                    type=config["problem_type"])
        end_time = time.time()
        wandb.log({"gan_loss": gan_loss, "fold": 1, "attack_type": "bruteForce"})
        print('Finished training in',end_time-train_time," seconds.")
        wandb.log({"training_time": end_time-train_time})
        op_path = (config['op_path'] + config['wandb_run'] + "/")
        syn_data = sample_data(synthesizer, config, op_path, train_df, data_prep, cv=True)
        eval(train_df, test_df, syn_data, op_path, cv=True, binary=True)
        wandb.finish()
        
    else:
        config = args.__dict__
        config.update({
            "problem_type": {"Classification": 'label'},
            "random_dim": 200,
            "epochs":100,
            "num_channels": 128,
            "eps": 0.00001,
            "lr": 0.0004969483674705974,
            "batch_size": 1000,
            "lambda_": 30,
            "seed": 23,
            "ip_path": "thesisgan/input/new_train_data.csv",
            "op_path": "thesisgan/output/",
            "weight_decay": 0.00000215862158190158,
            "wandb_run": "CTABGAN_binary_1",
        })
        print("Training non-hpo model - binary classification 1")
        train_data = pd.read_csv(config["ip_path"])
        test_data = pd.read_csv(config["test_data"])
        
        cols = list(train_data.columns)
        cols.remove("label")
        cols.append("label")
        train_data = train_data[cols]
        test_data = test_data[cols]
        
        attack_type_le = {"benign": 0, "bruteForce": 1, "portScan": 2, "pingScan": 3, "dos": 4}
        proto_le = {"TCP": 0, "UDP": 1, "ICMP": 2, "IGMP": 3}
        label_type_le = {"normal": 0, "attack": 1, "attacker": 1, "victim": 1}
        tos_le = {0 : 0, 32 : 1, 192 : 2, 16 : 3}
        #based on the unique values in the dataset, we will create a dictionary to map the values to integers
        datasets = [train_data, test_data]
        for dataset in datasets:
            dataset["attack_type"] = dataset["attack_type"].map(attack_type_le)
            dataset["proto"] = dataset["proto"].map(proto_le)
            dataset["tos"] = dataset["tos"].map(tos_le)
            dataset["label"] = dataset["label"].map(label_type_le)
                
        #convert args to a dictionary
        wandb.init(project="masterthesis",
                    config=config,
                    group="CTABGAN_binary", 
                    notes=config["desc"],
                    mode="offline",
                    name=config["wandb_run"])
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
        syn_data = sample_data(synthesizer, config, op_path, train_data, data_prep=data_prep)
        eval(train_data, test_data, syn_data, op_path, cv=False, binary=True)
        wandb.finish()
        
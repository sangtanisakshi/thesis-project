"""CLI."""
import sys
sys.path.append(".")               
import argparse
import pandas as pd
import numpy as np
import torch
import wandb
import optuna
import joblib
from optuna.samplers import TPESampler
from types import SimpleNamespace
import time
import os
import pickle
import glob
import plotly.io as pio
from synthesizers.ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder
from ctabganplus.model.evaluation import get_utility_metrics,stat_sim
from sklearn.model_selection import cross_validate
from thesisgan.model_evaluation import eval_model
from PIL import Image
import json
pio.renderers.default = 'iframe'
os.environ['WANDB_OFFLINE'] = 'true'

def _parse_args(discrete_columns):
    # add argument hpo to the parser and if hpo is True, add the hpo_config to the parser
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-name','--wandb_run', type=str, default="CTGAN_CV", help='Wandb run name')
    parser.add_argument('-desc','--description', type=str, default="CV 5 fold Using trial 12 params", help='Wandb run description')
    parser.add_argument('-op','--output', type=str, default='thesisgan/output/ctgan_cv/', help='Path of the output file')
    parser.add_argument('-test','--test_data', type=str, default='thesisgan/input/new_hpo_data.csv', help='Path to test data')
    parser.add_argument('-s','--seed', type=int, default=23, help='Random seed')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of training epochs')
    parser.add_argument('-d', '--discrete_columns', default=discrete_columns, help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-ver','--verbose', type=bool, default=True, help='Verbose')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help='If model should be saved')
    parser.add_argument('--load', action=argparse.BooleanOptionalAction, help='If model should be loaded')
    parser.add_argument('--model_path', default="thesisgan/output/ctgan_1/model.pkl", type=str, help='Path to the model file')
    parser.add_argument('-n', '--num-samples', type=int, default=None, help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument('--sample_condition_column', default=None, type=str, help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str, help='Specify the value of the selected discrete column.')
    parser.add_argument('-dsteps','--discriminator_steps', type=int, default=1, help='Discriminator steps')
    parser.add_argument('-ip','--data', type=str, default='thesisgan/input/new_train_data.csv', help='Path to training data')
    parser.add_argument('-glr', '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.')
    parser.add_argument('-dlr', '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.')
    parser.add_argument('-gdecay', '--generator_decay', type=float, default=1e-6, help='Weight decay for the generator.')
    parser.add_argument('-ddecay', '--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.')
    parser.add_argument('-edim', '--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.')
    parser.add_argument('-gdim', '--generator_dim', type=str, default=(256,256), help='Dimension of each generator layer.')
    parser.add_argument('-ddim', '--discriminator_dim', type=str, default=(256,256), help='Dimension of each discriminator layer.')
    parser.add_argument('-bs', '--batch_size', type=int, default=500,help='Batch size. Must be an even number.')
    parser.add_argument('-logfrq','--log_frequency', type=bool, default=True, help='Log frequency')
    parser.add_argument('--pac', type=int, default=10, help='PAC parameter')
    parser.add_argument('-l', '--lambda_', type=float, default=10, help='Gradient penalty lambda_ hyperparameter')
    parser.add_argument('-cuda', '--cuda', type=bool, default=True, help='Use GPU')
    parser.add_argument('--hpo', action=argparse.BooleanOptionalAction, help='Hyperparameter optimization mode')
    parser.add_argument('--cv', action=argparse.BooleanOptionalAction, help='Cross validation mode')
    return parser.parse_args()

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_hpo_parameters(trial):

    generator_lr = trial.suggest_float("generator_lr", 1e-4, 5e-4, log=True)
    discriminator_lr = trial.suggest_float("discriminator_lr", 1e-4, 5e-4, log=True)
    generator_decay = trial.suggest_float("generator_decay", 1e-6, 1e-4, log=True)
    discriminator_decay = trial.suggest_float("discriminator_decay", 1e-5, 1e-4, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    generator_dim = trial.suggest_categorical("generator_dim", [(256,256), (512,512), (1024,1024)])
    discriminator_dim = trial.suggest_categorical("discriminator_dim", [(256,256), (512,512), (1024,1024)])
    batch_size = trial.suggest_categorical("batch_size", [500, 1000, 2000])
    log_frequency= trial.suggest_categorical("log_frequency", [True, False])
    lambda_ = trial.suggest_categorical("lambda_", [10, 20, 30])
    
    hpo_params = dict({'generator_lr': generator_lr, 'discriminator_lr': discriminator_lr, 
                       'generator_decay': generator_decay, 'discriminator_decay': discriminator_decay, 
                       'embedding_dim': embedding_dim, 'generator_dim': generator_dim, 
                       'discriminator_dim': discriminator_dim, 'batch_size': batch_size, 
                       'log_frequency': log_frequency, 'lambda_': lambda_})
    print("Hyperparameters for current trial: ",hpo_params)
    return hpo_params

def wrapper(train_data, rest_args):
    def hpo(trial):
        print("Running trial number: ", trial.number, "out of 20")
        hpo_params = get_hpo_parameters(trial)
        config = {}
        config.update(hpo_params)
        config.update(rest_args)
        wandb.init(project="masterthesis", config=config, mode="offline", group="CTGAN_HPO_2", 
                   notes=config['description'])
        model = CTGAN(config)
        gan_loss = model.fit(train_data, config["discrete_columns"], config["epochs"])
        wandb.log({"WGAN-GP_experiment": gan_loss, "trial": trial.number})
        # get the current sweep id and create an output folder for the sweep
        trial = str(trial.number)
        op_path = (config['output'] + config['wandb_run'] + "/" + trial + "/")
        test_data, sampled_data = sample(model, config, op_path, train_data)
        eval(train_data, test_data, sampled_data, op_path)
        wandb.finish()
        return gan_loss
    
    return hpo

def sample(model, args, op_path, train_data, cv=False):
        
        os.makedirs(op_path, exist_ok=True)
        print("Saving?", str(args["save"]))
        if args["save"]:
            CTGAN.save(model, str(op_path+"model.pkl"))
            pickle.dump(model, open(str(op_path+"pklmodel.pkl"), "wb"))
    
        num_samples = train_data.shape[0] if args["num_samples"] is None else args["num_samples"]

        if args["sample_condition_column"] is not None:
            assert args["sample_condition_column_value"] is not None

        sample_start = time.time()
        sampled = model.sample(
            num_samples,
            args["sample_condition_column"],
            args["sample_condition_column_value"])
        sample_end = time.time() - sample_start
        print(f"Sampling time: {sample_end} seconds")
        wandb.log({"sampling_time": sample_end})
    
        print("Data sampling complete. Saving synthetic data...")
        sampled.to_csv(str(op_path+"syn.csv"), index=False)
        print("Synthetic data saved. Check the output folder.") 

        return sampled
    
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
    "tcp_fin": {"sdtype": "categorical"},
    "tos": {"sdtype": "categorical"},
    "attack_type": {"sdtype": "categorical"},
    "label": {"sdtype": "categorical"}
    }
    }

    print("Unique Values:" , test_data["attack_type"].unique(), test_data["label"].unique(), 
                         sample_data["attack_type"].unique(), sample_data["label"].unique(),
                         train_data["attack_type"].unique(), train_data["label"].unique())
    
    attack_type_le = {"benign": 0, "bruteForce": 1, "portScan": 2, "pingScan": 3, "dos": 4}
    proto_le = {"TCP": 0, "UDP": 1, "ICMP": 2, "IGMP": 3}
    label_type_le = {"normal": 0, "attack": 1, "attacker": 1, "victim": 1}
    tos_le = {0 : 0, 32 : 1, 192 : 2, 16 : 3}

    #based on the unique values in the dataset, we will create a dictionary to map the values to integers
    datasets = [train_data, test_data, sample_data]
    for dataset in datasets:
        dataset["attack_type"] = dataset["attack_type"].map(attack_type_le)
        dataset["proto"] = dataset["proto"].map(proto_le)
        dataset["tos"] = dataset["tos"].map(tos_le)
        if cv==False:
            dataset["label"] = dataset["label"].map(label_type_le)
        
    cat_cols = ['proto', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'attack_type', 'label']
    for col in cat_cols:
        test_data[col] = test_data[col].astype(str)
        sample_data[col] = sample_data[col].astype(str)
    
    #Data Evaluation
    scores = eval_model(test_data, sample_data, eval_metadata, op_path)

    # #Save the evaluation results to wandb
    imgs = [np.asarray(Image.open(f)) for f in sorted(glob.glob(op_path + "*.png"))]
    wandb.log({"Evaluation": [wandb.Image(img) for img in imgs]})
    wandb.log(scores)
    
    print("Unique Values:" , test_data["attack_type"].unique(), test_data["label"].unique(), 
                         sample_data["attack_type"].unique(), sample_data["label"].unique(),
                         train_data["attack_type"].unique(), train_data["label"].unique())
    for col in cat_cols:
        test_data[col] = test_data[col].astype("int64")
        sample_data[col] = sample_data[col].astype("int64")
        train_data[col] = train_data[col].astype("int64")

    if (binary==False or cv==False):
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
    result_df, cr = get_utility_metrics(train_data,test_data,sample_data,"MinMax",model_dict, cv=cv, binary=binary)

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
      

def main():
    """CLI."""
    discrete_columns = [
    'label',
    'attack_type',
    'proto',
    'tos',
    'tcp_ack',
    'tcp_psh',
    'tcp_rst',
    'tcp_syn',
    'tcp_fin']
    
    args = _parse_args(discrete_columns=discrete_columns) 
    # Set device to GPU if available
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    seed_everything(args.seed)

    print(torch.cuda.get_device_name(0))
    if args.hpo:
        args.data = "thesisgan/input/new_hpo_data.csv"
        rest_args = {
        'wandb_run': args.wandb_run,
        'description': args.description,
        'output': args.output,
        'test_data': args.test_data,
        'seed': args.seed,
        'epochs': args.epochs,
        'discrete_columns': args.discrete_columns,
        'verbose': args.verbose,
        'save': args.save,
        'load': args.load,
        'model_path': args.model_path,
        'num_samples': args.num_samples,
        'sample_condition_column': args.sample_condition_column,
        'sample_condition_column_value': args.sample_condition_column_value,
        'discriminator_steps': args.discriminator_steps,
        'hpo': args.hpo,
        'cuda': args.cuda,
        'pac': args.pac,
        }
    else:
        config = args.__dict__

    train_data = pd.read_csv(args.data)

    print(args.load)
    if args.load:
        print("Loading model...")
        model = pickle.load(open(args.model_path, "rb"))
        test_data = pd.read_csv(args.test_data)
        sample_data = pd.read_csv("thesisgan/output/ctgan_sweep/ycoi93zy/syn.csv")
        op_path = "thesisgan/output/ctgan_sweep/ycoi93zy/"
        eval(test_data, sample_data, op_path)
        
    elif args.hpo:
        print("Training HPO model...")
        sampler = TPESampler(seed=123)
        study = optuna.create_study(study_name="ctgan_hpo", direction="minimize", sampler=sampler)
        study.optimize(wrapper(train_data, rest_args), n_trials=20)
        joblib.dump(study,("thesisgan/hpo_results/ctgan_study.pkl"))
        study_data = pd.DataFrame(study.trials_dataframe())
        study_data.to_csv("thesisgan/hpo_results/ctgan_study_results.csv")
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial params:")
        for key, value in study.best_params.items():
            print(" {}: {}".format(key, value))
        
    elif args.cv:
        
        print("Training with Cross Validation")
        # update args to set with the best hp of the model - trial 12
        config.update({
            "epochs":2,
            "generator_lr": 0.00010024734990379357,
            "discriminator_lr": 0.00029082495033222255,
            "generator_decay": 0.0000392349472109858,
            "discriminator_decay": 0.00001412239691533274,
            "embedding_dim": 128,
            "generator_dim": "256,256",
            "discriminator_dim": "1024,1024",
            "batch_size": 1000,
            "log_frequency": True,
            "lambda_": 20,
            "seed": 23,
            "train_data": "thesisgan/input/new_train_data.csv",
            "test_data": "thesisgan/input/new_hpo_data.csv",
            "output": "thesisgan/output/ctgan_cv/"})

        # First we will convert the training data label to a binary classification algorithm
        # where normal traffic is 0 and attack traffic is 1
        #move the label at the end of the dataframe
        cols = list(train_data.columns)
        cols.remove("label")
        cols.append("label")
        train_data = train_data[cols]
        label_type_le = {"normal": 0, "attack": 1, "attacker": 1, "victim": 1}
        train_data["label"] = train_data["label"].map(label_type_le)
        # do a 4 fold cross validation where every fold is one type of attack.
        # we will first split the data into 5 folds based on the attack type
        # then we will train the model on 4 folds and evaluate on the 5th fold
        # we will repeat this process 5 times
        config["wandb_run"] = (f"CTGAN_CV_debug")
        wandb.init(project="masterthesis", name=config["wandb_run"], config=config, notes=config["description"],
                            group="CTGAN-CV-debug", mode="offline")
        print(f"Starting fold 4")
        test_df = train_data[train_data["attack_type"] == "bruteForce"]
        train_df = train_data[train_data["attack_type"] != "bruteForce"]
        model = CTGAN(config)
        gan_loss = model.fit(train_df, discrete_columns, config["epochs"])
        wandb.log({"gan_loss": gan_loss, "fold": 4, "attack_type": "bruteForce"})
        op_path = (config["output"] + config["wandb_run"] + "/")
        sampled_data = sample(model, config, op_path, train_df, cv=True)
        eval(train_df, test_df, sampled_data, op_path, cv=True, binary=True)
        wandb.finish()
    else:
        config.update({
        
        "epochs":300,
        "generator_lr": 0.00010024734990379357,
        "discriminator_lr": 0.00029082495033222255,
        "generator_decay": 0.0000392349472109858,
        "discriminator_decay": 0.00001412239691533274,
        "embedding_dim": 128,
        "generator_dim": "256,256",
        "discriminator_dim": "1024,1024",
        "batch_size": 1000,
        "log_frequency": True,
        "lambda_": 20,
        "seed": 23,
        "train_data": "thesisgan/input/new_train_data.csv",
        "test_data": "thesisgan/input/new_hpo_data.csv",
        "wandb_run": "CTGAN_binary_classification",
        "output": "thesisgan/output/"})
        # First we will convert the training data label to a binary classification algorithm
        # where normal traffic is 0 and attack traffic is 1
        test_data = pd.read_csv(config["test_data"])
        binary = True
        if binary:
            train_data["label"] = train_data["label"].apply(lambda x: 0 if x == "normal" else 1)
            #move the label at the end of the dataframe
            cols = list(train_data.columns)
            cols.remove("label")
            cols.append("label")
            train_data = train_data[cols]
        wandb.init(project="masterthesis", name=config["wandb_run"], config=config, notes=config["description"],
                                group="CTGAN-BINARY", mode="offline")
        print("Training model...")
        model = CTGAN(config)
        gan_loss = model.fit(train_df, discrete_columns, config["epochs"])
        wandb.log({"gan_loss": gan_loss})
        op_path = (config["output"] + config["wandb_run"] + "/")
        sampled_data = sample(model, config, op_path, train_df, cv=True)
        eval(train_df, test_df, sampled_data, op_path, cv=True, binary=True)
        wandb.finish()
if __name__ == '__main__':
    main()

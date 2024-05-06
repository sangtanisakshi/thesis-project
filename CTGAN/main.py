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
from thesisgan.model_evaluation import eval_model
from PIL import Image
import json
pio.renderers.default = 'iframe'
os.environ['WANDB_OFFLINE'] = 'true'

def _parse_args(discrete_columns):
    # add argument hpo to the parser and if hpo is True, add the hpo_config to the parser
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-name','--wandb_run', type=str, default="CTGAN_HPO", help='Wandb run name')
    parser.add_argument('-desc','--description', type=str, default="CTGAN_HPO_OPTUNA", help='Wandb run description')
    parser.add_argument('-op','--output', type=str, default='thesisgan/output/', help='Path of the output file')
    parser.add_argument('-test','--test_data', type=str, default='thesisgan/input/new_test_data.csv', help='Path to test data')
    parser.add_argument('-s','--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-e', '--epochs', default=150, type=int, help='Number of training epochs')
    parser.add_argument('-d', '--discrete_columns', default=discrete_columns, help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-ver','--verbose', type=bool, default=True, help='Verbose')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help='If model should be saved')
    parser.add_argument('--load', action=argparse.BooleanOptionalAction, help='If model should be loaded')
    parser.add_argument('--model_path', default="thesisgan/output/ctgan_1/model.pkl", type=str, help='Path to the model file')
    parser.add_argument('-n', '--num-samples', type=int, default=None, help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument('--sample_condition_column', default=None, type=str, help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str, help='Specify the value of the selected discrete column.')
    parser.add_argument('-dsteps','--discriminator_steps', type=int, default=1, help='Discriminator steps')
    parser.add_argument('-ip','--data', type=str, default='thesisgan/input/new_hpo_data.csv', help='Path to training data')
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
        hpo_params = get_hpo_parameters(trial)
        config = {}
        config.update(hpo_params)
        config.update(rest_args)
        wandb.init(project="masterthesis", config=config, mode="offline", group="CTGAN_HPO_OPTUNA", notes=config['description'])
        model = CTGAN(config)
        gan_loss = model.fit(train_data, config["discrete_columns"], config["epochs"])
        wandb.log({"WGAN-GP_experiment": gan_loss})
        # get the current sweep id and create an output folder for the sweep
        trial = str(trial.number)
        op_path = (config['output'] + config['wandb_run'] + "/" + trial + "/")
        test_data, sampled_data = sample(model, wandb.config, op_path)
        eval(train_data, test_data, sampled_data, op_path)
        return gan_loss
    
    return hpo

def sample(model, args, op_path):
        
        os.makedirs(op_path, exist_ok=True)
        print(f"Saving? {args.save}")
        if args.save:
            CTGAN.save(model, str(op_path+"model.pkl"))
            pickle.dump(model, open(str(op_path+"pklmodel.pkl"), "wb"))
        
        # Load test data
        test_data = pd.read_csv(args.test_data)
        if test_data.columns[0] == "Unnamed: 0":
            test_data = test_data.drop(columns="Unnamed: 0")
        print("Test data loaded")
    
        num_samples = test_data.shape[0] if args.num_samples is None else args.num_samples

        if args.sample_condition_column is not None:
            assert args.sample_condition_column_value is not None

        sample_start = time.time()
        sampled = model.sample(
            num_samples,
            args.sample_condition_column,
            args.sample_condition_column_value)
        sample_end = time.time() - sample_start
        print(f"Sampling time: {sample_end} seconds")
        wandb.log({"sampling_time": sample_end})
    
        print("Data sampling complete. Saving synthetic data...")
        sampled.to_csv(str(op_path+"syn.csv"), index=False)
        print("Synthetic data saved. Check the output folder.") 

        return test_data, sampled
    
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
    "tcp_fin": {"sdtype": "categorical"},
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
    
    le_dict = {"attack_type": "le_attack_type", "label": "le_label", "proto": "le_proto", "tos": "le_tos"}
    for c in le_dict.keys():
        le_dict[c] = LabelEncoder()
        test_data[c] = le_dict[c].fit_transform(test_data[c])
        sample_data[c] = le_dict[c].fit_transform(sample_data[c])
        train_data[c] = le_dict[c].fit_transform(train_data[c])
    
    for col in cat_cols:
        test_data[col] = test_data[col].astype("int64")
        sample_data[col] = sample_data[col].astype("int64")
        train_data[col] = train_data[col].astype("int64")
        
        #if the synthetic data does not have the same attack types as the test data, we need to fix it
        
    for at in test_data["attack_type"].unique():
        if at not in sample_data["attack_type"].unique():
            #add a row with the attack type
            sample_data = pd.concat([sample_data,train_data[train_data["attack_type"] == at].sample(3)], ignore_index=True)
    
    #Model Classification
    model_dict =  {"Classification":["lr","dt","rf","mlp"]}
    result_df, cr = get_utility_metrics(test_data,sample_data,"MinMax", model_dict, test_ratio = 0.30)

    stat_res_avg = []
    stat_res = stat_sim(test_data, sample_data, cat_cols)
    stat_res_avg.append(stat_res)

    stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
    stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
    
    wandb.log({'Stat Results' : wandb.Table(dataframe=stat_results), 
               'Classification Results': wandb.Table(dataframe=result_df),
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
        wandb_logging = wandb.init(project="masterthesis", name=args.wandb_run, config=args, notes=args.description, group="CTGAN")
        config = {
        'embedding_dim': args.embedding_dim,
        'generator_dim': args.generator_dim,
        'discriminator_dim': args.discriminator_dim,
        'generator_lr': args.generator_lr,
        'generator_decay': args.generator_decay,
        'discriminator_lr': args.discriminator_lr,
        'discriminator_decay': args.discriminator_decay,
        'batch_size': args.batch_size,
        'discriminator_steps': args.discriminator_steps,
        'log_frequency': args.log_frequency,
        'verbose': args.verbose,
        'epochs': args.epochs,
        'pac': args.pac,
        'lambda_' : args.lambda_,
        }

    train_data = pd.read_csv(args.data)

    print(args.load)
    if args.load:
        print("Loading model...")
        model = pickle.load(open(args.model_path, "rb"))
        test_data = pd.read_csv(args.test_data)
        sample_data = pd.read_csv("thesisgan/output/ctgan_sweep/ycoi93zy/syn.csv")
        op_path = "thesisgan/output/ctgan_sweep/ycoi93zy/"
        eval(test_data, sample_data, op_path)
    else:
        if args.hpo:
            print("Training HPO model...")
            sampler = TPESampler(seed=42)
            study = optuna.create_study(study_name="ctgan_hpo", direction="minimize", sampler=sampler)
            study.optimize(wrapper(train_data, rest_args), n_trials=15)
            joblib.dump(study,("thesisgan/hpo_results/ctgan_study.pkl"))
            study_data = pd.DataFrame(study.trials_dataframe())
            data_csv = study_data.to_csv("thesisgan/hpo_results/ctgan_study_results.csv")
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial params:")
            for key, value in study.best_params.items():
                print(" {}: {}".format(key, value))
    
            #get best trial hyperparameters and train the model with that
            best_params = SimpleNamespace(**study.best_params)
        else:
            print("Training model...")
            model = CTGAN(wandb.config)
            gan_loss = model.fit(train_data, discrete_columns, args.epochs)
            wandb.log({"gan_loss": gan_loss})

if __name__ == '__main__':
    main()

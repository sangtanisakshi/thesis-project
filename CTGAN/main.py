"""CLI."""
import sys
sys.path.append(".")               
import argparse
import pandas as pd
import numpy as np
import torch
import wandb
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

def _parse_args(discrete_columns):
    # add argument hpo to the parser and if hpo is True, add the hpo_config to the parser
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-name','--wandb_run', type=str, default="ctgan_sweep_debug_joblib", help='Wandb run name')
    parser.add_argument('-desc','--description', type=str, default="hpo joblib debug", help='Wandb run description')
    parser.add_argument('-op','--output', type=str, default='thesisgan/output/', help='Path of the output file')
    parser.add_argument('-test','--test_data', type=str, default='thesisgan/input/test_data.csv', help='Path to test data')
    parser.add_argument('-s','--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-e', '--epochs', default=1, type=int, help='Number of training epochs')
    parser.add_argument('-d', '--discrete_columns', default=discrete_columns, help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-ver','--verbose', type=bool, default=True, help='Verbose')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help='If model should be saved')
    parser.add_argument('--load', action=argparse.BooleanOptionalAction, help='If model should be loaded')
    parser.add_argument('--model_path', default="thesisgan/output/ctgan_1/model.pkl", type=str, help='Path to the model file')
    parser.add_argument('-n', '--num-samples', type=int, default=None, help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument('--sample_condition_column', default=None, type=str, help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str, help='Specify the value of the selected discrete column.')
    parser.add_argument('-dsteps','--discriminator_steps', type=int, default=1, help='Discriminator steps')
    parser.add_argument('-ip','--data', type=str, default='thesisgan/input/hpo_data.csv', help='Path to training data')
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

def wrapper(train_data, discrete_columns, epochs, rest_args):
    def hpo():
        wandb.init(project="masterthesis", config=wandb.config, group="CTGAN", notes=rest_args['description'])
        wandb.config.update(rest_args)
        model = CTGAN(wandb.config)
        gan_loss = model.fit(train_data, discrete_columns, epochs)
        wandb.log({"WGAN-GP_experiment": gan_loss})
        # get the current sweep id and create an output folder for the sweep
        sweep_id = wandb.run.id
        op_path = (rest_args['output'] + rest_args['wandb_run'] + "/" + sweep_id + "/")
        test_data, sampled_data = sample(model, wandb.config, op_path)
        eval(test_data, sampled_data, op_path)
        
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
    
def eval(test_data, sample_data, op_path):
    eval_metadata = {
    "columns" : 
    {
    "duration": {"sdtype": "numerical", "compute_representation": "Float" },
    "proto": {"sdtype": "categorical"},
    "src_pt": {"sdtype": "numerical", "compute_representation": "Float"},
    "dst_pt": {"sdtype": "numerical", "compute_representation": "Float"},
    "packets": {"sdtype": "numerical", "compute_representation": "Float"},
    "bytes": {"sdtype": "numerical", "compute_representation": "Float"},
    "tcp_ack": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_psh": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_rst": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_syn": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_fin": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tos": {"sdtype": "numerical", "compute_representation": "Int64"},
    "label": {"sdtype": "categorical"},
    "attack_type": {"sdtype": "categorical"}
    }
    }
    test_data.drop(columns=["tcp_urg"], inplace=True)
    sample_data.drop(columns=["tcp_urg"], inplace=True)
    
    #remove columns with only one unique value
    for col in test_data.columns:
        if len(test_data[col].unique()) == 1:
            test_data.drop(columns=[col], inplace=True)
        if len(sample_data[col].unique()) == 1:
            sample_data.drop(columns=[col], inplace=True)
    
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
        test_data[c] = test_data[c].astype("int64")
        sample_data[c] = le_dict[c].fit_transform(sample_data[c])
        sample_data[c] = sample_data[c].astype("int64")
    
    #Model Classification
    model_dict =  {"Classification":["lr","dt","rf","mlp"]}
    real_results, fake_results, results_diff = get_utility_metrics(test_data, sample_data,"MinMax", model_dict, test_ratio = 0.30)

    # get real, fake and diff results and put the acc, auc and f1 scores in a dataframe
    diff_df = pd.DataFrame(results_diff,columns=["Acc","AUC","F1_Score"])
    diff_df.index = list(model_dict.values())[0]
    diff_df.index.name = "Model"
    diff_df["Model"] = diff_df.index
    diff_df["Type"] = "Difference"
    
    real_df = pd.DataFrame(real_results,columns=["Acc","AUC","F1_Score"])
    real_df.index = list(model_dict.values())[0]
    real_df.index.name = "Model"
    real_df["Model"] = real_df.index
    real_df["Type"] = "Real"
    
    fake_df = pd.DataFrame(fake_results,columns=["Acc","AUC","F1_Score"])
    fake_df.index = list(model_dict.values())[0]
    fake_df.index.name = "Model"
    fake_df["Model"] = fake_df.index
    fake_df["Type"] = "Fake"
    
    #concatenate the dataframes
    result_df = pd.concat([real_df,fake_df,diff_df])
    result_df = result_df.reset_index(drop=True)

    stat_res_avg = []
    stat_res = stat_sim(test_data, sample_data, list(le_dict.keys()))
    stat_res_avg.append(stat_res)

    stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
    stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
    wandb.log({'Stat Results' : wandb.Table(dataframe=stat_results), 'Classification Results': wandb.Table(dataframe=result_df)})
    print("Evaluation complete. Check the output folder for the plots and evaluation results.")
      

def main():
    """CLI."""
    discrete_columns = [
    'label',
    'attack_type',
    'proto',
    'tos',
    'tcp_con',
    'tcp_ech',
    'tcp_urg',
    'tcp_ack',
    'tcp_psh',
    'tcp_rst',
    'tcp_syn',
    'tcp_fin' ]
    
    args = _parse_args(discrete_columns=discrete_columns) 
    # Set device to GPU if available
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    seed_everything(args.seed)

    print(torch.cuda.get_device_name(0))
    wandb.login()
    if args.hpo:
        args.data = "thesisgan/input/hpo_data.csv"
        hpo_config = {
        'method': 'bayes', 
        'metric': {'name': 'gan_loss', 'goal': 'minimize'},
        'parameters': {
            'generator_lr': {'values': [2e-4, 1e-4, 5e-4]},
            'discriminator_lr': {'values': [2e-4, 1e-4, 5e-4]},
            'generator_decay': {'values': [1e-6, 1e-5, 1e-4]},
            'discriminator_decay': {'values': [0, 1e-5, 1e-4]},
            'embedding_dim': {'values': [128, 256, 512]},
            'generator_dim': {'values': [(256,256), (512,512), (1024,1024)]},
            'discriminator_dim': {'values': [(256,256), (512,512), (1024,1024)]},
            'batch_size': {'values': [500, 1000, 2000]},
            'log_frequency': {'values': [True, False]},
            'lambda_': {'values': [10, 20, 30]}
        }
        }
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
            sweep_id = wandb.sweep(sweep=hpo_config, project="masterthesis")
            wandb.agent(sweep_id, function=wrapper(train_data, args.discrete_columns, args.epochs, rest_args), count=3, project="masterthesis")
        else:
            print("Training model...")
            model = CTGAN(wandb.config)
            gan_loss = model.fit(train_data, discrete_columns, args.epochs)
            wandb.log({"gan_loss": gan_loss})

if __name__ == '__main__':
    main()

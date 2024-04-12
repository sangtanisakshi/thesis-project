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
from synthesizers.ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder
from ctabganplus.model.evaluation import get_utility_metrics,stat_sim
from thesisgan.model_evaluation import eval_model
from PIL import Image

def _parse_args():
    
    discrete_columns = [
    'label',
    'attack_type',
    'attack_id',
    'proto',
    'day_of_week',
    'tos',
    'tcp_con',
    'tcp_ech',
    'tcp_urg',
    'tcp_ack',
    'tcp_psh',
    'tcp_rst',
    'tcp_syn',
    'tcp_fin' ]
    
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV')
    parser.add_argument('--no-header', dest='header', action='store_false', help='The CSV file has no header. Discrete columns will be indices.')
    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument('-d', '--discrete_columns', default=discrete_columns, help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-n', '--num-samples', type=int, default=10000, help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument('-glr', '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.')
    parser.add_argument('-dlr', '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.')
    parser.add_argument('-gdecay', '--generator_decay', type=float, default=1e-6, help='Weight decay for the generator.')
    parser.add_argument('-ddecay', '--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.')
    parser.add_argument('-edim', '--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.')
    parser.add_argument('-gdim', '--generator_dim', type=str, default=(256,256), help='Dimension of each generator layer.')
    parser.add_argument('-ddim', '--discriminator_dim', type=str, default=(256,256), help='Dimension of each discriminator layer.')
    parser.add_argument('-bs', '--batch_size', type=int, default=500,help='Batch size. Must be an even number.')
    parser.add_argument('-dsteps','--discriminator_steps', type=int, default=1, help='Discriminator steps')
    parser.add_argument('-logfrq','--log_frequency', type=bool, default=True, help='Log frequency')
    parser.add_argument('-ver','--verbose', type=bool, default=True, help='Verbose')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help='If model should be saved')
    parser.add_argument('--load', action=argparse.BooleanOptionalAction, help='If model should be loaded')
    parser.add_argument('--model_path', default="thesisgan/output/ctgan_1/model.pkl", type=str, help='Path to the model file')
    parser.add_argument('--sample_condition_column', default=None, type=str, help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str, help='Specify the value of the selected discrete column.')
    parser.add_argument('--pac', type=int, default=10, help='PAC parameter')
    parser.add_argument('-name','--wandb_run', type=str, default="ctgan_4", help='Wandb run name')
    parser.add_argument('-desc','--description', type=str, default="debug", help='Wandb run description')
    parser.add_argument('-ip','--data', type=str, default='thesisgan/input/train_data.csv', help='Path to training data')
    parser.add_argument('-op','--output', type=str, default='thesisgan/output/', help='Path of the output file')
    parser.add_argument('-test','--test_data', type=str, default='thesisgan/input/test_data.csv', help='Path to test data')
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    print(args.load)
    logging = wandb.init(project="masterthesis", name=args.wandb_run, config=args, notes=args.description, group="CTGAN")
    
    # Set device to GPU if available
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    print(torch.cuda.get_device_name(0))

    train_data = pd.read_csv(args.data)

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
    'pac': args.pac
    }
    
    op_path = (args.output + args.wandb_run + "/")
    os.makedirs(op_path, exist_ok=True)
    print(args.load)
    if args.load:
        print("Loading model...")
        model = pickle.load(open(args.model_path, "rb"))
    else:
        print("Training model...")
        model = CTGAN(**config)
        model.fit(train_data, args.discrete_columns)
    
    print(f"Saving? {args.save}")
    if args.save:
        CTGAN.save(model, str(op_path+"model.pkl"))
        pickle.dump(model, open(str(op_path+"model.pkl"), "wb"))

    # Load test data
    test_data = pd.read_csv(args.test_data)
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
    
    eval_metadata = {
    "columns" : 
    {
    "time_of_day": {"sdtype": "numerical", "compute_representation": "Float"},
    "duration": {"sdtype": "numerical", "compute_representation": "Float" },
    "proto": {"sdtype": "categorical"},
    "src_pt": {"sdtype": "numerical", "compute_representation": "Float"},
    "dst_pt": {"sdtype": "numerical", "compute_representation": "Float"},
    "packets": {"sdtype": "numerical", "compute_representation": "Float"},
    "bytes": {"sdtype": "numerical", "compute_representation": "Float"},
    "tcp_con": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_ech": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_ack": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_psh": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_rst": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_syn": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tcp_fin": {"sdtype": "numerical", "compute_representation": "Int64"},
    "tos": {"sdtype": "categorical"},
    "label": {"sdtype": "categorical"},
    "attack_type": {"sdtype": "categorical"},
    "attack_id": {"sdtype": "categorical"},
    "day_of_week": {"sdtype": "categorical"},
    "src_ip_1": {"sdtype": "numerical", "compute_representation": "Float"},
    "src_ip_2": {"sdtype": "numerical", "compute_representation": "Float"},
    "src_ip_3": {"sdtype": "numerical", "compute_representation": "Float"},
    "src_ip_4": {"sdtype": "numerical", "compute_representation": "Float"},
    "dst_ip_1": {"sdtype": "numerical", "compute_representation": "Float"},
    "dst_ip_2": {"sdtype": "numerical", "compute_representation": "Float"},
    "dst_ip_3": {"sdtype": "numerical", "compute_representation": "Float"},
    "dst_ip_4": {"sdtype": "numerical", "compute_representation": "Float"}
    }
    }
    
    #Data Evaluation
    scores = eval_model("ctgan", test_data, sampled, eval_metadata, op_path)
    
    # Save the evaluation results to wandb
    imgs = [np.asarray(Image.open(f)) for f in sorted(glob.glob(op_path + "*.png"))]
    wandb.log({"Evaluation": [wandb.Image(img) for img in imgs]})
    wandb.log(scores)
    
    le_dict = {"attack_type": "le_attack_type", "label": "le_label", "proto": "le_proto", "day_of_week": "le_day_of_week", "tos": "le_tos"}
    for c in le_dict.keys():
        le_dict[c] = LabelEncoder()
        test_data[c] = le_dict[c].fit_transform(test_data[c])
        test_data[c] = test_data[c].astype("int64")
        sampled[c] = le_dict[c].fit_transform(sampled[c])
        sampled[c] = sampled[c].astype("int64")
    
    #Model Classification
    model_dict =  {"Classification":["lr","dt","rf","mlp","svm"]}
    result_mat = get_utility_metrics(test_data, sampled,"MinMax", model_dict, test_ratio = 0.30)

    result_df  = pd.DataFrame(result_mat,columns=["Acc","AUC","F1_Score"])
    result_df.index = list(model_dict.values())[0]
    result_df.index.name = "Model"
    result_df["Model"] = result_df.index
    
    stat_res_avg = []
    stat_res = stat_sim(test_data, sampled, list(le_dict.keys()))
    stat_res_avg.append(stat_res)

    stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
    stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
    logging.log({'Stat Results' : wandb.Table(dataframe=stat_results), 'Classification Results': wandb.Table(dataframe=result_df)})
    
    print("Evaluation complete. Check the output folder for the plots and evaluation results.")
    
if __name__ == '__main__':
    main()

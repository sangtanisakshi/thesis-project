"""CLI."""

import argparse
import pandas as pd
import torch
import wandb
import time
from ctgan.synthesizers.ctgan import CTGAN
import os
from thesisGAN.model_evaluation import eval_model

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
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument('-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV')
    parser.add_argument('--no-header', dest='header', action='store_false', help='The CSV file has no header. Discrete columns will be indices.')
    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument('-d', '--discrete_columns', default=discrete_columns, help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-n', '--num-samples', type=int, default=None, help='Number of rows to sample. Defaults to the training data size')
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
    parser.add_argument('-ver','--verbose', type=bool, default=False, help='Verbose')
    parser.add_argument('--save', default=False, type=bool, help='If model should be saved')
    parser.add_argument('--load', default=None, type=str, help='A filename to load a trained synthesizer.')
    parser.add_argument('--sample_condition_column', default=None, type=str, help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str, help='Specify the value of the selected discrete column.')
    parser.add_argument('--pac', type=int, default=10, help='PAC parameter')
    parser.add_argument('-name','--wandb_run', type=str, default="ctgan_run_1", help='Wandb run name')
    parser.add_argument('-desc','--description', type=str, default="test_run", help='Wandb run description')
    parser.add_argument('-ip','--data', type=str, default='thesisGAN/input/trainval_data.csv', help='Path to training data')
    parser.add_argument('-op','--output', type=str, default='thesisGAN/output/', help='Path of the output file')

    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    
    # Set device to GPU if available
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    print(torch.cuda.get_device_name(0))
    
    wandb.init(project="masterthesis", name=args.wandb_run, config=args, notes=args.description, group="CTGAN")

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
    
    if args.load:
        model = CTGAN.load(args.load)
    else:
        model = CTGAN(**config)
    model.fit(train_data, args.discrete_columns)
    
    if args.save:
        model.save(op_path)

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
    sampled.to_csv(str(op_path+"_syn.csv"), index=False)
    print("Synthetic data saved. Check the output folder.")
    
    print("Running data evaluation")
    eval_model("ctgan",  test_data, sampled, op_path)
    
    # Save the evaluation results to wandb
    wandb.save(str(op_path+"*"))

if __name__ == '__main__':
    main()

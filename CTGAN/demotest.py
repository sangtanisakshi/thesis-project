from ctgan.synthesizers.ctgan_m import CTGAN
import pandas as pd
import torch
import wandb
import argparse

# Set device to GPU if available
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

print(torch.cuda.get_device_name(0))

#Default discrete columns
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
'tcp_fin']

# Get input arguments for the path of the input data, the hyperparameters for the config data
parser = argparse.ArgumentParser(description='CTGAN')
parser.add_argument('-ip','--input_data', type=str, default='thesisGAN/input/train_data.csv', help='Path to the input data')
parser.add_argument('-dc','--discrete_columns', type=list, default=discrete_columns, help='Discrete columns')
parser.add_argument('-wb','--wandb_run', type=str, default="test_run", help='Wandb run number')
parser.add_argument('-emdim','--embedding_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('-gdim','--generator_dim', type=tuple, default=(256, 256), help='Generator dimensions')
parser.add_argument('-ddim','--discriminator_dim', type=tuple, default=(256, 256), help='Discriminator dimensions')
parser.add_argument('-glr','--generator_lr', type=float, default=2e-4, help='Generator learning rate')
parser.add_argument('-gdecay','--generator_decay', type=float, default=1e-6, help='Generator decay')
parser.add_argument('-dlr','--discriminator_lr', type=float, default=2e-4, help='Discriminator learning rate')
parser.add_argument('-ddecay','--discriminator_decay', type=float, default=1e-6, help='Discriminator decay')
parser.add_argument('-bsize','--batch_size', type=int, default=500, help='Batch size')
parser.add_argument('-dsteps','--discriminator_steps', type=int, default=1, help='Discriminator steps')
parser.add_argument('-logfrq','--log_frequency', type=bool, default=True, help='Log frequency')
parser.add_argument('-ver','--verbose', type=bool, default=False, help='Verbose')
parser.add_argument('-e','--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--pac', type=int, default=10, help='PAC parameter')
parser.add_argument('--cuda', type=bool, default=True, help='Use GPU')

parser.add_argument('-s','--samples', type=int, default=10000, help='Number of samples to generate')
args = parser.parse_args()

# Config dict
config = {
    'discrete_columns': args.discrete_columns,
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
    'cuda': args.cuda,
}

wandb.init(project="masterthesis", name=str("CTGAN_",args.wandb_run), config=config)

# Load dataset
real_data = pd.read_csv(args.input_data)

ctgan = CTGAN(**config)

ctgan.fit(real_data, discrete_columns)

# Create synthetic data
print("Model training complete. Sampling data...")
synthetic_data = ctgan.sample(args.samples)
print(synthetic_data.head(20))

synthetic_data.to_csv('thesisGAN/model-outputs/ctgan_synthetic_data.csv', index=False)

wandb.log({"synthetic_data": wandb.Table(dataframe=synthetic_data)})

print("Synthetic data saved. Check the output folder.")

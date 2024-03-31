import wandb
import pandas as pd
import torch
import numpy as np
import argparse
import sdmetrics
import sdgym

parser = argparse.ArgumentParser(description='Model Evaluation CLI')

# Get the synthetic data
parser.add_argument('-ctgan','--ctgan_syn_data', type=str, default='thesisGAN/model-outputs/ctgan_syn_data.csv', help='Path of the CTGAN synthetic data')
parser.add_argument('-itgan','--itgan_syn_data', type=str, default='thesisGAN/model-outputs/itgan_syn_data.csv', help='Path of the ITGAN synthetic data')
parser.add_argument('-ctabgan','--ctabgan_syn_data', type=str, default='thesisGAN/model-outputs/ctabgan_syn_data.csv', help='Path of the CTABGAN synthetic data')

# Get the real data
parser.add_argument('-real','--real_data', type=str, default='thesisGAN/input/test_data.csv', help='Path of the real data')

# Import all models and the data for evaluation


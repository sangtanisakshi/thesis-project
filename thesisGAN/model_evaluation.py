import wandb
import pandas as pd
import torch
import numpy as np
import argparse
from table_evaluator import TableEvaluator

parser = argparse.ArgumentParser(description='Model Evaluation CLI')

# Get the synthetic data
parser.add_argument('-ctgan','--ctgan_syn_data', type=str, default='model-outputs/ctgan_syn_data.csv', help='Path of the CTGAN synthetic data')
parser.add_argument('-itgan','--itgan_syn_data', type=str, default='model-outputs/itgan_syn_data.csv', help='Path of the ITGAN synthetic data')
parser.add_argument('-ctabgan','--ctabgan_syn_data', type=str, default='model-outputs/ctabgan_syn_data.csv', help='Path of the CTABGAN synthetic data')
parser.add_argument('-m','--model', type=str, default='ctgan', help='Model to evaluate')
# Get the real data
parser.add_argument('-real','--real_data', type=str, default='input/test_data.csv', help='Path of the real data')

args = parser.parse_args()

real_data = pd.read_csv(args.real_data)

if args.model == 'ctgan':
    syn_data = pd.read_csv(args.ctgan_syn_data)
elif args.model == 'itgan':
    syn_data = pd.read_csv(args.itgan_syn_data)
elif args.model == 'ctabgan':
    syn_data = pd.read_csv(args.ctabgan_syn_data)
    
real_data.drop(columns=["tcp_urg"], inplace=True)
syn_data.drop(columns=["tcp_urg"], inplace=True)

# Statistical Similarity

save_dir = ('evaluation_outputs/evaluator/'+args.model+'/')
table_eval = TableEvaluator(real_data, syn_data, verbose=True)
table_eval.visual_evaluation(save_dir)

# SDMetrics

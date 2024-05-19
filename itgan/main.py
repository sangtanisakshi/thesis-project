import argparse
import sys
from types import SimpleNamespace
import joblib
sys.path.append(".")    
import os
import glob
import wandb
os.environ["WANDB_OFFLINE"] = "true"
os.environ["WANDB_ERROR_REPORTING"] = "false"
import pickle
import torch
import time
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import plotly.io as pio
pio.renderers.default = 'iframe'
from PIL import Image
import warnings
warnings.filterwarnings(action='ignore')
import optuna
from optuna.samplers import TPESampler

from ctabganplus.model.evaluation import get_utility_metrics, stat_sim
from thesisgan.model_evaluation import eval_model
from util.data import load_dataset
from util.model_test import mkdir
from util.model import AEGANSynthesizer
from synthesizers.AEModel import Encoder, Decoder, loss_function
from synthesizers.GeneratorModel import Generator, argument
from synthesizers.DiscriminatorModel import Discriminator
from sklearn.preprocessing import LabelEncoder
import requests

session = requests.Session()
session.trust_env = False

def parse_args():
    parser = argparse.ArgumentParser('ITGAN')
    parser.add_argument('--wandb_run', type=str, help= 'Run name', default="ITGAN_HPO_LOCAL")
    parser.add_argument('-desc', '--wb_desc', type=str, help= 'Run description', default="Itgan HPO Optuna Run")
    parser.add_argument('--data', type=str, default = 'itgan_debug')
    parser.add_argument('--op_path', type=str, default = 'thesisgan/output/')
    parser.add_argument('--seed', type=int, default = 42)
    parser.add_argument('--hpo', default = False)
    parser.add_argument('--save', default = True)
    parser.add_argument('--epochs',type =int, default = 50)  
    parser.add_argument('--n_trials', type=int, default = 8)
    parser.add_argument('--num_samples', type=int, default = None)
    parser.add_argument('--emb_dim', type=int, default = "128") # dim(h)
    parser.add_argument('--en_dim', type=str, default = "256,128") # n_e(r) = 2 -> "256,128", 3 -> "512,256,128" 
    parser.add_argument('--d_dim', type=str, default = "256,256") # n_d = 2 -> "256,256", 3 -> "256,256,256" 
    parser.add_argument('--d_dropout', type=float, default= 0.5) # a
    parser.add_argument('--d_leaky', type=float, default=0.2) # b
    parser.add_argument('--hdim_factor', type=float, default=1.) # M
    parser.add_argument('--likelihood_coef', type=float, default=0) # gamma
    parser.add_argument('--gt', type=int, default = 1) # period_G
    parser.add_argument('--dt', type=int, default = 1) # period_D
    parser.add_argument('--lt', type=int, default = 6) # period_L
    args = parser.parse_args()
    
    return args

def seed_everything(random_num=23):
    
    torch.manual_seed(random_num)
    torch.cuda.manual_seed(random_num)
    torch.cuda.manual_seed_all(random_num) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_num)

def get_hpo_parameters(trial):
    emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 128]) #dim(h)
    en_dim = trial.suggest_categorical("en_dim", [(256,128), (512,256,128)]) #n_e(r)
    d_dim = trial.suggest_categorical("d_dim", [(256,256), (256, 256, 256)]) #n_d
    d_dropout = trial.suggest_categorical("D_dropout", [0, 0.5]) #dropout rate a
    d_leaky = trial.suggest_categorical("D_leaky", [0, 0.2]) #leaky relu slope b
    hdim_factor = trial.suggest_categorical("hdim_factor", [1., 1.5]) #M
    likelihood_coef = trial.suggest_categorical("likelihood_coef", [-0.1, -0.014, -0.012, -0.01, 0, 0.01, 0.014, 0.05, 0.1]) #gamma
    gt = trial.suggest_categorical("gt", [1, 3, 5, 6]) #period_G
    dt = trial.suggest_categorical("dt", [1, 3, 5, 6]) #period_D
    lt = trial.suggest_categorical("lt", [1, 3, 5, 6]) #period_L
    
    decompress_dims = en_dim[::-1]
    
    hpo_params = dict({"compress_dims": en_dim, "decompress_dims": decompress_dims,
                       "embedding_dim": emb_dim, "dis_dim": d_dim, 
                       "D_dropout": d_dropout, "D_leaky": d_leaky, "hdim_factor": hdim_factor, 
                       "likelihood_coef": likelihood_coef, "G_learning_term": gt, "D_learning_term": dt, "likelihood_learn_term": lt})
    print("Hyperparameters for current trial: ",hpo_params)
    return hpo_params

def wrapper(arg, G_args, train_df, test_df, meta, categoricals, ordinals):
    def hpo(trial):
        hpo_params = get_hpo_parameters(trial)
        config = {}
        config.update(hpo_params)
        G_args["hdim_factor"] = float(hpo_params["hdim_factor"])
        config.update(arg)
        logging.info("Config: ", config)
        logging.info("Started trial new ")
        wb_run = wandb.init(project="masterthesis", config=config, mode="offline",
                            group="itgan_hpo_debug", notes=config['description'])
        logging.info("wandb initialized")
        config["G_args"] = argument(G_args, hpo_params["embedding_dim"])
        synthesizer = AEGANSynthesizer(config)
        logging.info("Synthesizer created")
        gan_loss = synthesizer.fit(train_df, test_df, meta, config["data_name"], categoricals, ordinals)
        wandb.log({"WGAN-GP_experiment": gan_loss})
        # get the current sweep id and create an output folder for the sweep
        trial = str(trial.number)
        op_path = (config['save_loc'] + config['wandb_run'] + "/" + trial + "/")
        test_data, sampled_data = sample(synthesizer, test_df, config, op_path)
        eval(train_df, test_data, sampled_data, op_path, trial)
        wb_run.finish()
        logging.debug("Finished trial")
        return gan_loss
    
    return hpo

def sample(model, test_data, args, op_path):
        
        os.makedirs(op_path, exist_ok=True)
        print("Saving?" + args["save"])
        if args["save"]:
            pickle.dump(model, open(str(op_path+"model.pkl"), "wb"))
            
        num_samples = test_data.shape[0] if args["num_samples"] is None else args["num_samples"]

        sample_start = time.time()
        sampled = model.sample(
            num_samples)
        sample_end = time.time() - sample_start
        print("Sampling time:", sample_end ,"seconds")
        wandb.log({"sampling_time": sample_end})
        
        df_columns = ['duration', 'proto', 'src_pt', 'dst_pt', 'packets',
       'bytes', 'tcp_ack', 'tcp_psh','tcp_rst', 'tcp_syn', 'tcp_fin', 
       'tos','label','attack_type']
        syn_data = pd.DataFrame(sampled, columns=df_columns)
        test_data = pd.DataFrame(test_data, columns=df_columns)
        
        print("Data sampling complete. Saving synthetic data...")
        syn_data.to_csv(str(op_path+"syn.csv"), index=False)
        print("Synthetic data saved. Check the output folder.") 
        print("Syn Data", syn_data.info())
        print("Test_data", test_data.info())
        return test_data, syn_data
    
def eval(train_data, test_data, sample_data, op_path, trial):
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

    df_columns = ['duration', 'proto', 'src_pt', 'dst_pt', 'packets',
    'bytes', 'tcp_ack', 'tcp_psh','tcp_rst', 'tcp_syn', 'tcp_fin', 
    'tos','label','attack_type']
    train_data = pd.DataFrame(train_data, columns=df_columns)
    
    print(train_data.info())
    #remove columns with only one unique value
    for col in test_data.columns:
        if len(test_data[col].unique()) == 1:
            print("Removing column, ", col, " as it has only one unique value")
            test_data.drop(columns=[col], inplace=True)
            sample_data.drop(columns=[col], inplace=True)

    le_dict = {"attack_type": "le_attack_type", "label": "le_label", "proto": "le_proto", "tos": "le_tos"}
    for c in le_dict.keys():
        le_dict[c] = LabelEncoder()
        test_data[c] = le_dict[c].fit_transform(test_data[c])
        sample_data[c] = le_dict[c].fit_transform(sample_data[c])
        train_data[c] = le_dict[c].fit_transform(train_data[c])
        test_data[c] = test_data[c].astype("int64")
        sample_data[c] = sample_data[c].astype("int64")
        train_data[c] = train_data[c].astype("int64")
        
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
    
    print(test_data.attack_type.unique(), sample_data.attack_type.unique(), train_data.attack_type.unique())
    
    #if the synthetic data does not have the same attack types as the test data, we need to fix it
    for at in test_data.attack_type.unique():
        if at not in sample_data.attack_type.unique():
            #add a row with the attack type
            print(at)
            sample_data = pd.concat([sample_data,train_data[train_data["attack_type"] == at].sample(3)], ignore_index=True)
    
    #if the synthetic data has only one unique value for the attack type, add a row with the attack type
    sample_data_value_counts = sample_data["attack_type"].value_counts()
    for i in range(len(sample_data_value_counts)):
        if sample_data_value_counts[i] == 1:
            at = sample_data_value_counts.index[i]
            sample_data = pd.concat([sample_data,train_data[train_data["attack_type"] == at].sample(3)], ignore_index=True)
            
    cat_cols = ['proto', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'label', 'attack_type']
    for col in cat_cols:
        test_data[col] = test_data[col].astype("float64")
        test_data[col] = test_data[col].astype("int64")
        sample_data[col] = sample_data[col].astype("float64")
        sample_data[col] = sample_data[col].astype("int64")
        train_data[col] = train_data[col].astype("float64")
        train_data[col] = train_data[col].astype("int64")
    
    
    #Model Classification
    model_dict =  {"Classification":["xgb","lr","dt","rf","mlp"]}
    result_df, cr = get_utility_metrics(train_data, test_data, sample_data,"MinMax", model_dict)

    stat_res_avg = []
    stat_res = stat_sim(test_data, sample_data, cat_cols)
    stat_res_avg.append(stat_res)

    stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
    stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
    
    result_df["run"] = wandb.run.id
    cr["run"] = wandb.run.id
    stat_results["run"] = wandb.run.id
    
    wandb.log({'Stat Results' : wandb.Table(dataframe=stat_results), 
               'Classification Results': wandb.Table(dataframe=result_df),
               'Classification Report': wandb.Table(dataframe=cr)})
    
    print("Evaluation complete. Check the output folder for the plots and evaluation results.")


# Commented when testing
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    print(torch.cuda.get_device_name(0))
    
    arg_of_parser = parse_args()
    seed_everything(arg_of_parser.seed)
    
    #Generator CNF Info
    G_args = {
            'layer_type' : "blend", # layer type ["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"] 
                                    #blend[M_i(z,t) = t], simblenddiv1[M_i(z,t) = sigmoid(FC(z⊕t))]
            'nhidden' : 3,
            'num_blocks' : 1, # the number of ode block(cnf statck) <int>
            'time_length' : 1.0, # time length(if blend type is choosed time length has to be 1.0) <float>
            'train_T' : False, # Traing T(if blend type is choosed this has to be False)  [True, False]
            'divergence_fn' : "approximate", # how to calculate jacobian matrix ["brute_force", "approximate"]
            'nonlinearity' : "tanh", # the act func to use # ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
            'solver' : "dopri5", # ode solver ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
            'atol' : 1e-3,
            'rtol' : 1e-3,
            'test_solver' : "dopri5", # ode solver ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
            'test_atol' : 1e-3,
            'test_rtol' : 1e-3,
            'step_size' : None, # "Optional fixed step size." <float, None>
            'first_step' : 0.166667, # only for adaptive solvers  <float> 
            'residual' : True,  # use residual net odefunction [True, False]
            'rademacher' : False, # rademacher or gaussian [True, False],
            'batch_norm' : False,  # use batch norm [True, False]
            'bn_lag' : 0., # batch_norm of bn_lag <float>
            # regularizer of odefunction(you must use either regularizer group 1 or regularizer group 2, not both)
            # regularizer group 1
            'l1int' : None,
            'l2int' : None, #"int_t ||f^T df/dt||_2"  <float, None>
            'dl2int' : None,  #"int_t ||df/dx||_F"  <float, None>
            'JFrobint' : None, # "int_t ||df/dx||_F"  <float, None>
            'JdiagFrobint' : None, # "int_t ||df_i/dx_i||_F"  <float, None>
            'JoffdiagFrobint' : None, # "int_t ||df/dx - df_i/dx_i||_F"  <float, None>
            # regularizer group 2
            'kinetic_energy' : 1., # int_t ||f||_2^2 <float, None> # kinetic regularizer coef
            'jacobian_norm2' : 1., # int_t ||df/dx||_F^2 <float, None>
            'total_deriv' : None, # int_t ||df/dt||^2 <float, None>
            'directional_penalty' : None,  # int_t ||(df/dx)^T f||^2 <float, None>
            'adjoint' : True
            } # use adjoint method for backpropagation [True, False]
    
    arg = {
            "rtol":1e-3,
            "atol":1e-3,
            "batch_size":4096,
            "random_num":23,
            "GPU_NUM":0,
            "G_model":Generator,
            "G_lr":2e-4,
            "G_beta":(0.5, 0.9),
            "G_l2scale":1e-6,
            "G_l1scale":0,
            "likelihood_learn_start_score":None,
            "kinetic_learn_every_G_learn":0, # if 0 apply kinetic regularizer every G likelihood training, else every all G training
            "D_model":Discriminator,
            "dis_dim":(256, 256),
            "lambda_grad":10,
            "D_lr":2e-4,
            "D_beta":(0.5, 0.9),
            "D_l2scale":1e-6,
            "En_model":Encoder,
            "AE_lr":2e-4,
            "AE_beta":(0.5, 0.9),
            "AE_l2scale":1e-6,
            'ae_learning_term': 1,
            'ae_learning_term_g' : 1,
            "De_model":Decoder,
            "L_func":loss_function,
            "loss_factor":2.,
            "n_trials" : arg_of_parser.n_trials,
            "epochs":arg_of_parser.epochs,
            "save_loc":arg_of_parser.op_path,
            "seed":arg_of_parser.seed,
            "wandb_run":arg_of_parser.wandb_run,
            "description":arg_of_parser.wb_desc,
            'hpo': arg_of_parser.hpo,
            'save': arg_of_parser.save,
            'num_samples': arg_of_parser.num_samples,
            
        }
    
    # if arg_of_parser.hpo:
    #     arg["data_name"] = "malware_hpo"
    
    # else:
    #     compress_dims = tuple([int(i) for i in arg_of_parser.en_dim.split(",")])
    #     decompress_dims = compress_dims[::-1]
    #     dis_dim = tuple([int(i) for i in arg_of_parser.d_dim.split(",")])
    
    #     arg.update({
    #         "data_name": "malware",
    #         "embedding_dim": arg_of_parser.emb_dim,
    #         "dis_dim": dis_dim,
    #         "G_learning_term" : arg_of_parser.gt,
    #         "likelihood_coef":arg_of_parser.likelihood_coef,
    #         "likelihood_learn_term":arg_of_parser.lt,
    #         'D_learning_term' : arg_of_parser.dt,
    #         'D_leaky' : arg_of_parser.d_leaky,
    #         'D_dropout': arg_of_parser.d_dropout,
    #         "compress_dims":compress_dims,
    #         "decompress_dims":decompress_dims
    #         })
        
    #     # Generator CNF info
    #     G_args.update({
    #         'hdim_factor' : arg_of_parser.hdim_factor})
        
    #     arg["G_args"] = argument(G_args, arg["embedding_dim"])
    #     arg["save_arg"] = arg.copy() 
        
    if arg_of_parser.hpo:
        train_df, test_df, meta, categoricals, ordinals = load_dataset(arg["data_name"], benchmark=True)
        print("Data Loaded")
        logging.basicConfig(filename='it/test_gc_log.log', level=logging.DEBUG)
        logging.debug("HPO started")
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=123))
        study.optimize(wrapper(arg, G_args, train_df, test_df, meta, categoricals, ordinals), n_trials=arg_of_parser.n_trials, gc_after_trial=True)
        print("HPO complete. Check the output folder")
        joblib.dump(study,(f"thesisgan/hpo_results/itgan_hpo_study_{arg_of_parser.wandb_run}.pkl"))
        study_data = pd.DataFrame(study.trials_dataframe())
        data_csv = study_data.to_csv(f"thesisgan/hpo_results/itgan_study_results_{arg_of_parser.wandb_run}.csv")
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial params:")
        for key, value in study.best_params.items():
            print(" {}: {}".format(key, value))
    
        #get best trial hyperparameters and train the model with that
        best_params = SimpleNamespace(**study.best_params)
    else:
        hpo_params = {
            'compress_dims': (512, 256, 128), 
            'decompress_dims': (128, 256, 512), 
            'embedding_dim': 128, 
            'dis_dim': (256, 256, 256), 
            'D_dropout': 0.5, 
            'D_leaky': 0.2, 
            'hdim_factor': 1.5, 
            'likelihood_coef': -0.1, 
            'G_learning_term': 5, 
            'D_learning_term': 6, 
            'likelihood_learn_term': 1
        }
        arg.update({"data_name": "malware_hpo"})
        config = {}
        config.update(hpo_params)
        train_df, test_df, meta, categoricals, ordinals = load_dataset(arg["data_name"], benchmark=True)
        print("Data Loaded")
        logging.basicConfig(filename='it/manual_1_20ep_2.log', level=logging.DEBUG)
        logging.debug("HPO started manually")
        G_args["hdim_factor"] = float(hpo_params["hdim_factor"])
        config.update(arg)
        logging.info("Config: ", config)
        logging.info("Started trial new ")
        wb_run = wandb.init(project="masterthesis", config=config, mode="offline",
                            group="itgan_manual_hpo_20ep", notes=config['description'])
        logging.info("wandb initialized")
        config["G_args"] = argument(G_args, hpo_params["embedding_dim"])
        synthesizer = AEGANSynthesizer(config)
        logging.info("Synthesizer created")
        gan_loss = synthesizer.fit(train_df, test_df, meta, config["data_name"], categoricals, ordinals)
        wandb.log({"WGAN-GP_experiment": gan_loss, "trial": "1"})
        # get the current sweep id and create an output folder for the sweep
        op_path = (config['save_loc'] + config['wandb_run'] + "/1_20ep/" + "/")
        test_data, sampled_data = sample(synthesizer, test_df, config, op_path)
        eval(train_df, test_data, sampled_data, op_path, "1")
        wb_run.finish()
        logging.debug("Finished trial")

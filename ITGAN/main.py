import argparse
import sys
sys.path.append("..")    
import os
import glob
import wandb
import pickle
import numpy as np
import pandas as pd
import plotly.io as pio
pio.renderers.default = 'iframe'
from PIL import Image
import warnings
warnings.filterwarnings(action='ignore')

from torch.nn import functional as F

from ctabganplus.model.evaluation import get_utility_metrics, stat_sim
from thesisgan.model_evaluation import eval_model

from util.data import load_dataset
from util.evaluate import compute_scores
from util.evaluate_cluster import compute_cluster_scores
from util.model_test import mkdir
from sklearn.preprocessing import LabelEncoder

from synthesizers.AEModel import Encoder, Decoder, loss_function
from synthesizers.GeneratorModel import Generator, argument
from synthesizers.DiscriminatorModel import Discriminator
import pickle
from util.model import AEGANSynthesizer

def train(synthesizer, syn_arg, dataset):
    """
    Train the model and compute last scores of Clustering and scores of Supervised Learning
    """
    #synthesizer = synthesizer(**syn_arg)
    synthesizer = pickle.load(open("../thesisgan/output/itgan_1/model.pkl", "rb"))
    train, test, meta, categoricals, ordinals = load_dataset(dataset, benchmark=True)
    print("Data Loaded")
    #synthesizer.fit(train, test, meta, dataset, categoricals, ordinals)
    synthesized = synthesizer.sample(test.shape[0])
    print("Data sampled")
    # scores = compute_scores(train, test, synthesized, meta)
    # if 'likelihood' in meta["problem_type"]:
    #     return scores
    # scores_cluster = compute_cluster_scores(train, test, synthesized, meta)
    return synthesizer, synthesized, test


# Commented when testing
if __name__ == "__main__":
    ################################################ Default Value #######################################################
    ## basic info
    
    abspath = os.path.abspath(__file__)
    #dname = os.path.dirname(abspath)
    #os.chdir(dname)
    #print(os.getcwd())
    
    rtol = 1e-3 ; atol = 1e-3; batch_size = 2000 ; epochs = 1 ; random_num = 777 ; GPU_NUM = 0 ; 
    save_loc= "../thesisgan/output/itgan_1/"
    G_model= Generator; embedding_dim= 128; G_lr= 2e-4; G_beta= (0.5, 0.9); G_l2scale= 1e-6 ; G_l1scale = 0 ; 
    G_learning_term = 3 ; 
    likelihood_coef = 0 ; likelihood_learn_start_score = None ; likelihood_learn_term = 6 ; 
    kinetic_learn_every_G_learn = False

    D_model = Discriminator; 
    dis_dim= (256, 256); 
    lambda_grad= 10;
    D_lr= 2e-4; 
    D_beta= (0.5, 0.9);
    D_l2scale= 1e-6
    D_learning_term = 1 ; 
    D_leaky = 0.2 ; D_dropout = 0.5
    
    En_model= Encoder; compress_dims= (256, 128); AE_lr= 2e-4; AE_beta= (0.5, 0.9); 
    AE_l2scale= 1e-6 ; ae_learning_term = 1 ; ae_learning_term_g = 1
    De_model= Decoder; decompress_dims= (128, 256); L_func= loss_function; loss_factor = 2

    # Generator CNF info
    layer_type = "blend" # layer type ["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
    hdim_factor = 1. # hidden layer size <int>
    nhidden = 3 # the number of hidden layers <int>
    num_blocks = 1 # the number of ode block(cnf statck) <int>
    time_length = 1.0 # time length(if blend type is choosed time length has to be 1.0) <float>
    train_T = False # Traing T(if blend type is choosed this has to be False)  [True, False]
    divergence_fn = "approximate" # how to calculate jacobian matrix ["brute_force", "approximate"]
    nonlinearity = "tanh" # the act func to use # ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
    test_solver = solver = "dopri5" # ode solver ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
    test_atol = atol  # <float>
    test_rtol = rtol  # <float>
    step_size = None # "Optional fixed step size." <float, None>
    first_step = 0.166667 # only for adaptive solvers  <float> 
    residual = True  # use residual net odefunction [True, False]
    rademacher = False # rademacher or gaussian [True, False]
    batch_norm = False  # use batch norm [True, False]
    bn_lag = 0. # batch_norm of bn_lag <float>
    adjoint = True

    # regularizer of odefunction(you must use either regularizer group 1 or regularizer group 2, not both)
    # regularizer group 1
    l1int = None # "int_t ||f||_1" <float, None>
    l2int = None # "int_t ||f||_2" <float, None>
    dl2int = None # "int_t ||f^T df/dt||_2"  <float, None>
    JFrobint = None # "int_t ||df/dx||_F"  <float, None>
    JdiagFrobint = None # int_t ||df_i/dx_i||_F  <float, None>
    JoffdiagFrobint = None # int_t ||df/dx - df_i/dx_i||_F  <float, None>

    # regularizer group 2
    kinetic_energy = 1. # int_t ||f||_2^2 <float, None>
    jacobian_norm2 = 1. # int_t ||df/dx||_F^2 <float, None>
    total_deriv = None # int_t ||df/dt||^2 <float, None>
    directional_penalty = None  # int_t ||(df/dx)^T f||^2 <float, None>
    ################################################ Default Value #######################################################
    # python train_itgan.py --data --random_num --GPU_NUM --emb_dim --en_dim --d_dim --d_dropout --d_leaky --layer_type --hdim_factor --nhidden --likelihood_coef --gt --dt --lt --kinetic
    parser = argparse.ArgumentParser('ITGAN')
    parser.add_argument('-name', '--wb_name', type=str, help= 'Run name', default="itgan_1")
    parser.add_argument('-desc', '--wb_desc', type=str, help= 'Run description', default="Debugging the main file")
    parser.add_argument('--data', type=str, default = 'malware')
    parser.add_argument('--epochs',type =int, default = 1)  
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--GPU_NUM', type = int, default = 0)

    parser.add_argument('--emb_dim', type=int, default = 128) # dim(h)
    parser.add_argument('--en_dim', type=str, default = "256,128") # n_e(r) = 2 -> "256,128", 3 -> "512,256,128" 
    parser.add_argument('--d_dim', type=str, default = "256,256") # n_d = 2 -> "256,256", 3 -> "256,256,256" 

    parser.add_argument('--d_dropout', type=float, default= 0.5) # a
    parser.add_argument('--d_leaky', type=float, default=0.2) # b
    
    parser.add_argument('--layer_type', type=str, default="blend") # blend[M_i(z,t) = t], simblenddiv1[M_i(z,t) = sigmoid(FC(z⊕t))]
    parser.add_argument('--hdim_factor', type=float, default=1.) # M
    parser.add_argument('--nhidden', type=int, default = 3) # K
    parser.add_argument('--likelihood_coef', type=float, default=0) # gamma
    parser.add_argument('--gt', type=int, default = 1) # period_G
    parser.add_argument('--dt', type=int, default = 1) # period_D
    parser.add_argument('--lt', type=int, default = 6) # period_L

    parser.add_argument('--kinetic', type=float, default = 1.)  # kinetic regularizer coef
    parser.add_argument('--kinetic_every_learn', type=int, default = 0) # if 0 apply kinetic regularizer every G likelihood training, else every all G training
    
    
    arg_of_parser = parser.parse_args()
    data = arg_of_parser.data
    epochs = arg_of_parser.epochs 
    random_num = arg_of_parser.random_num
    GPU_NUM = arg_of_parser.GPU_NUM
    
    embedding_dim = arg_of_parser.emb_dim
    compress_dims = tuple([int(i) for i in arg_of_parser.en_dim.split(",")])
    decompress_dims = compress_dims[::-1]
    dis_dim = tuple([int(i) for i in arg_of_parser.d_dim.split(",")])

    D_leaky = arg_of_parser.d_leaky ; D_dropout = arg_of_parser.d_dropout

    layer_type = arg_of_parser.layer_type
    hdim_factor = arg_of_parser.hdim_factor
    nhidden = arg_of_parser.nhidden
    likelihood_coef = arg_of_parser.likelihood_coef
    G_learning_term = arg_of_parser.gt
    D_learning_term = arg_of_parser.dt
    likelihood_learn_term = arg_of_parser.lt
    kinetic_energy = jacobian_norm2 = arg_of_parser.kinetic
    kinetic_learn_every_G_learn = arg_of_parser.kinetic_every_learn
    
    test_name = ("_".join([str(i) for i in vars(arg_of_parser).values() if str(i) != data])).replace(" ", "")
    
    G_args = {
        'layer_type' : layer_type,
        'hdim_factor' : hdim_factor,
        'nhidden' : nhidden,
        'num_blocks' : num_blocks,
        'time_length' : time_length,
        'train_T' : train_T,
        'divergence_fn' : divergence_fn,
        'nonlinearity' : nonlinearity,
        'solver' : solver,
        'atol' : atol,
        'rtol' : rtol,
        'test_solver' : test_solver,
        'test_atol' : test_atol,
        'test_rtol' : test_rtol,
        'step_size' : step_size,
        'first_step' : first_step,
        'residual' : residual,
        'rademacher' : rademacher,
        'batch_norm' : batch_norm,
        'bn_lag' : bn_lag,
        'l1int' : l1int,
        'l2int' : l2int,
        'dl2int' : dl2int,
        'JFrobint' : JFrobint,
        'JdiagFrobint' : JdiagFrobint,
        'JoffdiagFrobint' : JoffdiagFrobint,
        'kinetic_energy' : kinetic_energy,
        'jacobian_norm2' : jacobian_norm2,
        'total_deriv' : total_deriv,
        'directional_penalty' : directional_penalty,
        'adjoint' : adjoint}

    arg = {"rtol":rtol,
            "atol":atol,
            "batch_size":batch_size,
            "epochs":epochs,
            "random_num":random_num,
            "GPU_NUM":GPU_NUM,
            "save_loc":save_loc,
            "test_name":test_name,
            "data_name": data,
            "G_model":G_model,
            "embedding_dim":embedding_dim,
            "G_lr":G_lr,
            "G_beta":G_beta,
            "G_l2scale":G_l2scale,
            "G_l1scale":G_l1scale,
            "G_learning_term" : G_learning_term,
            "likelihood_coef":likelihood_coef,
            "likelihood_learn_start_score":likelihood_learn_start_score,
            "likelihood_learn_term":likelihood_learn_term,
            "kinetic_learn_every_G_learn":kinetic_learn_every_G_learn,
            "D_model":D_model,
            "dis_dim":dis_dim,
            "lambda_grad":lambda_grad,
            "D_lr":D_lr,
            "D_beta":D_beta,
            "D_l2scale":D_l2scale,
            'D_learning_term' : D_learning_term,
            'D_leaky' : D_leaky,
            'D_dropout': D_dropout,
            "En_model":En_model,
            "compress_dims":compress_dims,
            "AE_lr":AE_lr,
            "AE_beta":AE_beta,
            "AE_l2scale":AE_l2scale,
            'ae_learning_term': ae_learning_term,
            'ae_learning_term_g' : ae_learning_term_g,
            "De_model":De_model,
            "decompress_dims":decompress_dims,
            "L_func":L_func,
            "loss_factor":loss_factor}
    
    wbrun = wandb.init(project='masterthesis', name=arg_of_parser.wb_name, config=arg, group="itgan", notes=arg_of_parser.wb_desc)
    
    # Log the G args as a table
    wandb.config.update(G_args)
    
    arg["G_args"] = argument(G_args, embedding_dim)
    arg["save_arg"] = arg.copy()   
    mkdir(save_loc, data)
    
    with open(save_loc + "/param/"+ data + "/" + test_name + '.txt',"w") as f:
        f.write(data + " AEGANSynthesizer" + "\n")
        f.write(str(arg) + "\n")
        f.write(str(G_args) + "\n")
    

    model, syn_data, test_data = train(AEGANSynthesizer, arg, data)
    #pickle.dump(model, open(str(save_loc+"/model.pkl"), "wb"))
    
    df_columns = ['time_of_day', 'duration', 'proto', 'src_pt', 'dst_pt', 'packets',
       'bytes', 'tcp_con', 'tcp_ech', 'tcp_urg', 'tcp_ack', 'tcp_psh',
       'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'attack_type', 'attack_id',
       'day_of_week', 'src_ip_1', 'src_ip_2', 'src_ip_3', 'src_ip_4',
       'dst_ip_1', 'dst_ip_2', 'dst_ip_3', 'dst_ip_4', 'label']
    syn_data = pd.DataFrame(syn_data, columns=df_columns)
    test_data = pd.DataFrame(test_data, columns=df_columns)
    
    syn_data.to_csv(save_loc + "syn_data.csv", index=False)
    
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
    scores = eval_model(test_data, syn_data, eval_metadata, save_loc)
    
    # # Save the evaluation results to wandb
    # imgs = [np.asarray(Image.open(f)) for f in sorted(glob.glob(save_loc + "*.png"))]
    # wandb.log({"Evaluation": [wandb.Image(img) for img in imgs]})
    # wandb.log(scores)
    
    le_dict = {"attack_type": "le_attack_type", "label": "le_label", "proto": "le_proto", "day_of_week": "le_day_of_week", "tos": "le_tos"}
    
    #Model Classification
    model_dict =  {"Classification":["lr","dt","rf","mlp","svm"]}
    result_mat = get_utility_metrics(test_data, syn_data,"MinMax", model_dict, test_ratio = 0.30)

    result_df  = pd.DataFrame(result_mat,columns=["Acc","AUC","F1_Score"])
    result_df.index = list(model_dict.values())[0]
    result_df.index.name = "Model"
    result_df["Model"] = result_df.index
    
    stat_res_avg = []
    stat_res = stat_sim(test_data, syn_data, list(le_dict.keys()))
    stat_res_avg.append(stat_res)

    stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
    stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
    wbrun.log({'Stat Results' : wandb.Table(dataframe=stat_results), 'Classification Results': wandb.Table(dataframe=result_df)})
    
    print("Evaluation complete. Check the output folder for the plots and evaluation results.")
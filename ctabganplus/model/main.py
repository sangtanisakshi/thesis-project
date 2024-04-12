
import numpy as np
import pandas as pd
import glob
import torch
import time
import warnings
import argparse
import wandb
import sys
sys.path.append(".")
from ctabgan import CTABGANSynthesizer
from evaluation import get_utility_metrics,stat_sim
from data_preparation import DataPrep
from sklearn.preprocessing import LabelEncoder
from thesisgan.model_evaluation import eval_model


def train(real_data, categorical_columns, log_columns, mixed_columns, general_columns, non_categorical_columns, integer_columns, problem_type, test_ratio, epochs, n, CTABGANSynthesizer):

    start_time = time.time()
    data_prep = DataPrep(real_data,categorical_columns,log_columns,mixed_columns,general_columns,non_categorical_columns,integer_columns,problem_type,test_ratio)
    CTABGANSynthesizer.fit(train_data=data_prep.df, categorical = data_prep.column_types["categorical"], mixed = data_prep.column_types["mixed"],
    general = data_prep.column_types["general"], non_categorical = data_prep.column_types["non_categorical"], type=problem_type)
    end_time = time.time()
    print('Finished training in',end_time-start_time," seconds.")
    sample = CTABGANSynthesizer.sample(n) 
    sample_df = data_prep.inverse_prep(sample)
        
    return sample_df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--categorical_columns", nargs='+', default=['attack_type','day_of_week','label','tos','proto'])
    parser.add_argument("--log_columns", nargs='+', default=[])
    parser.add_argument("--mixed_columns", nargs='+', default={}])
    parser.add_argument("--general_columns", nargs='+', default=[])
    parser.add_argument("--non_categorical_columns", nargs='+', default= ['packets','src_ip_1','src_ip_2','src_ip_3','src_ip_4',
                                'dst_ip_1','dst_ip_2','dst_ip_3','dst_ip_4','src_pt','dst_pt', 'time_of_day','duration','bytes'])
    parser.add_argument("--integer_columns", nargs='+', default=['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'])
    parser.add_argument("--problem_type", type=dict, default={"Classification": 'label'})
    parser.add_argument("--num_exp", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="Malware")
    parser.add_argument("--ip_path", type=str, default="thesisgan/input/trainval_data.csv")
    parser.add_argument("--op_path", type=str, default="thesisgan/output/")
    parser.add_argument("--test_data", type=str, default="thesisgan/input/test_data.csv")
    parser.add_argument("--wb_run", type=str, default="ctabgan_1")
    parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()

    real_data = pd.DataFrame(pd.read_csv(args.ip_path))

    #reorder columns in the raw_df such that the column "label" is the last column
    cols = real_data.columns.tolist()
    cols.remove('label')
    cols.append('label')
    raw_df = real_data[cols]

    columns = ["attack_type", "label", "proto", "day_of_week"]
    for c in columns:
        exec(f'le_{c} = LabelEncoder()')
        raw_df[c] = globals()[f'le_{c}'].fit_transform(raw_df[c])
        raw_df[c] = raw_df[c].astype("int64")

    for i in range(args.num_exp):
        train()

model_dict = {"Classification": ["lr", "dt", "rf", "mlp", "svm"]}
result_mat = get_utility_metrics(raw_df, fake_paths, "MinMax", model_dict, test_ratio=args.test_ratio)

result_df = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
result_df.index = list(model_dict.values())[0]

stat_res_avg = []
for fake_path in args.op_path:
    stat_res = stat_sim(raw_df, fake_path, args.categorical_columns)
    stat_res_avg.append(stat_res)

stat_columns = ["Average WD (Continuous Columns", "Average JSD (Categorical Columns)", "Correlation Distance"]
stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)
stat_results

if torch.cuda.is_available():
device = torch.device("cuda")
print(torch.cuda.get_device_name(0))
warnings.filterwarnings("ignore")

num_exp = 1
dataset = "Malware"
real_path = "Real_Datasets/trainval_data.csv"
fake_file_root = "Fake_Datasets"


synthesizer =  CTABGAN(raw_df,
                test_ratio = 0.77951,
                categorical_columns = ['attack_type','day_of_week','label','tos','proto'], 
                log_columns = [],
                mixed_columns= {},
                general_columns = [],
                non_categorical_columns = ['packets','src_ip_1','src_ip_2','src_ip_3','src_ip_4','dst_ip_1','dst_ip_2',
                                        'dst_ip_3','dst_ip_4','src_pt','dst_pt', 'time_of_day','duration','bytes'],
                integer_columns = ['attack_id','tcp_con','tcp_ech','tcp_urg','tcp_ack','tcp_psh','tcp_rst','tcp_syn','tcp_fin'],
                problem_type= {"Classification": 'label'},
                synthesizer = CTABGANSynthesizer(epochs=10))


syn.to_csv(fake_file_root+"/"+dataset+"/"+ dataset+"_fake_{exp}.csv".format(exp=i), index= False)

fake_paths = glob.glob(fake_file_root+"/"+dataset+"/"+"*")

model_dict =  {"Classification":["lr","dt","rf","mlp","svm"]}
result_mat = get_utility_metrics(raw_df,fake_paths,"MinMax",model_dict, test_ratio = 0.20)

result_df  = pd.DataFrame(result_mat,columns=["Acc","AUC","F1_Score"])
result_df.index = list(model_dict.values())[0]

malware_categorical = ['attack_type','day_of_week','label','tos','proto']
stat_res_avg = []
for fake_path in fake_paths:
stat_res = stat_sim(raw_df,fake_path,malware_categorical)
stat_res_avg.append(stat_res)

stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
stat_results
from model.ctabgan import CTABGAN
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from model.evaluation import get_utility_metrics, stat_sim, privacy_metrics
import numpy as np
import pandas as pd
import glob
import argparse
import torch
from model.data_preparation import DataPrep
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from model.synthesizer.ctabgan_synthesizer import Sampler, Cond, Generator, weights_init, determine_layers_gen

# Turn the following into inputs for argparse
num_exp = 1
dataset = "Malware"
real_path = "Real_Datasets/test_data.csv"
fake_file_root = "Fake_Datasets"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_exp", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="Malware")
    parser.add_argument("--real_path", type=str, default="Real_Datasets/test_data.csv")
    parser.add_argument("--fake_file_root", type=str, default="Fake_Datasets")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # print(torch.cuda.get_device_name(0))

    # synthesizer =  CTABGAN(raw_csv_path = real_path,
    #                 test_ratio = 0.30,
    #                 categorical_columns = ['proto', 'attack_type','day_of_week','label','tos'],
    #                 log_columns = [],
    #                 mixed_columns= {},
    #                 general_columns = [],
    #                 non_categorical_columns = ['packets','src_ip_1','src_ip_2','src_ip_3','src_ip_4','dst_ip_1','dst_ip_2',
    #                                             'dst_ip_3','dst_ip_4','src_pt','dst_pt', 'time_of_day','duration','bytes'],
    #                 integer_columns = ['attack_id','tcp_con','tcp_ech','tcp_urg','tcp_ack','tcp_psh','tcp_rst','tcp_syn','tcp_fin'],
    #                 problem_type= {"Classification": 'attack_type'},
    #                 synthesizer = CTABGANSynthesizer(epochs=1)
    #                 )

    # for i in range(num_exp):
    #     synthesizer.fit()
    categorical_columns = ["proto", "attack_type", "day_of_week", "label", "tos"]
    log_columns = []
    mixed_columns = {}
    general_columns = []
    non_categorical_columns = [
            "packets",
            "src_ip_1",
            "src_ip_2",
            "src_ip_3",
            "src_ip_4",
            "dst_ip_1",
            "dst_ip_2",
            "dst_ip_3",
            "dst_ip_4",
            "src_pt",
            "dst_pt",
            "time_of_day",
            "duration",
            "bytes",
        ]
    integer_columns = [
            "attack_id",
            "tcp_con",
            "tcp_ech",
            "tcp_urg",
            "tcp_ack",
            "tcp_psh",
            "tcp_rst",
            "tcp_syn",
            "tcp_fin",
        ]
    raw_df = pd.read_csv(real_path)
    raw_df['attack_id'] = raw_df['attack_id'].astype(int)
    raw_df['attack_id'] = raw_df['attack_id'].apply(lambda x: x - 17 if x != 0 else x)
    test_ratio = 0.1
    problem_type = {"Classification": "attack_type"}
    syn = CTABGANSynthesizer(epochs=1)
    train_data = DataPrep(
        raw_df,
        categorical_columns,
        log_columns,
        mixed_columns,
        general_columns,
        non_categorical_columns,
        integer_columns,
        problem_type,
        test_ratio,
    ).df
    # if problem_type:
    #     problem_type = list(problem_type.keys())[0]
    #     if problem_type:
    #         target_index = train_data.columns.get_loc(type[problem_type])
    syn.transformer = DataTransformer(
        train_data=train_data,
        categorical_list=categorical_columns,
        mixed_dict=mixed_columns,
        general_list=general_columns,
        non_categorical_list=non_categorical_columns,
    )
    print("Data transformed, now fitting the model")
    syn.transformer.fit()
    train_data = syn.transformer.transform(train_data.values)
    data_sampler = Sampler(train_data, syn.transformer.output_info)
    data_dim = syn.transformer.output_dim
    syn.cond_generator = Cond(train_data, syn.transformer.output_info)

    sides = [4, 8, 16, 24, 32, 64, 128, 256]
    col_size_d = data_dim + syn.cond_generator.n_opt
    for i in sides:
        if i * i >= col_size_d:
            syn.dside = i
            break

    sides = [4, 8, 16, 24, 32, 64, 128, 256]
    col_size_g = data_dim
    for i in sides:
        if i * i >= col_size_g:
            syn.gside = i
            break

    layers_G = determine_layers_gen(
        syn.gside, syn.random_dim + syn.cond_generator.n_opt, syn.num_channels
    )

    syn.generator = Generator(syn.gside, layers_G).to(syn.device)

    syn.generator.apply(weights_init)
    syn.Gtransformer = ImageTransformer(syn.gside)
    syn.Dtransformer = ImageTransformer(syn.dside)
    syn.sample(20)

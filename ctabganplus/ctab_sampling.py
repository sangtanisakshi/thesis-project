import pandas as pd
import numpy as np
import pickle
import json
import sys
sys.path.append('.')

from model.evaluation import get_utility_metrics
from sklearn.preprocessing import LabelEncoder
from model.data_preparation import DataPrep

# set seed 

ctabgan_best = pickle.load(open('thesisgan/output/ctabgan_best_model_2/pklmodel.pkl', 'rb'))
train_data = pd.read_csv('thesisgan/input/new_train_data.csv')
hpo_data = pd.read_csv("thesisgan/input/new_hpo_data.csv")
test_data = pd.read_csv("thesisgan/input/new_test_data.csv")
og_syn_data = pd.read_csv("thesisgan/output/ctabgan_best_model_2/syn.csv")

data_prep = DataPrep(train_data, ['proto', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin', 'tos', 'label', 'attack_type'], 
                     [], {}, [], ['packets', 'src_pt', 'dst_pt', 'duration', 'bytes'], [],
                     {"Classification": 'attack_type'}, 0.00003)

attack_type_le = {"benign": 0, "bruteForce": 1, "portScan": 2, "pingScan": 3, "dos": 4}
proto_le = {"TCP": 0, "UDP": 1, "ICMP": 2, "IGMP": 3}
label_type_le = {"normal": 0, "attack": 1, "attacker": 2, "victim": 3}
tos_le = {0 : 0, 32 : 1, 192 : 2, 16 : 3}

#based on the unique values in the dataset, we will create a dictionary to map the values to integers
datasets = [train_data, hpo_data, test_data, og_syn_data]
for dataset in datasets:
    dataset["attack_type"] = dataset["attack_type"].map(attack_type_le)
    dataset["proto"] = dataset["proto"].map(proto_le)
    dataset["tos"] = dataset["tos"].map(tos_le)
    dataset["label"] = dataset["label"].map(label_type_le)
    
more_samples = ctabgan_best.sample(train_data.shape[0])
og_more_samples = data_prep.inverse_prep(more_samples)


np.random.seed(42)
og_cresults, og_creport = get_utility_metrics(train_data, hpo_data, og_syn_data, scaler="MinMax", type={"Classification":["xgb","lr","dt","rf","mlp"]})
og = og_cresults.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
og["Type"] = og.index
og["Desc"] = "Original with hpo sized samples (train-hpo (hposized))"
og_creport["Desc"] = "Original with hpo sized samples (train-hpo (hposized))"

np.random.seed(42)
more_samples_results, more_samples_cr = get_utility_metrics(train_data, hpo_data, og_more_samples, scaler="MinMax",type={"Classification":["xgb","lr","dt","rf","mlp"]})
msr = more_samples_results.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
msr["Desc"] = "Original with train sized samples (train-hpo (trainsized))"
msr["Type"] = msr.index
more_samples_cr["Desc"] = "Original with train sized samples (train-hpo (trainsized))"

test_data = pd.concat([test_data, hpo_data])
test_data.reset_index(drop=True, inplace=True)

np.random.seed(42)
og_more_test_samples, og_more_test_cr = get_utility_metrics(train_data, test_data, og_syn_data, scaler="MinMax",type={"Classification":["xgb","lr","dt","rf","mlp"]})
og_more_test_samples_res = og_more_test_samples.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
og_more_test_samples_res["Desc"] = "Original with hpo sized samples, more test data (train- test+hpo))"
og_more_test_samples_res["Type"] = og_more_test_samples_res.index
og_more_test_cr["Desc"] = "Original with hpo sized samples, more test data (train- test+hpo))"

np.random.seed(42)
more_test_samples, more_test_cr = get_utility_metrics(train_data, test_data, og_more_samples, scaler="MinMax",type={"Classification":["xgb","lr","dt","rf","mlp"]})
more_test_samples_res = more_test_samples.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
more_test_samples_res["Desc"] = "Original with train sized samples, more test data (train- test+hpo))"
more_test_samples_res["Type"] = more_test_samples_res.index
more_test_cr["Desc"] = "Original with train sized samples, more test data (train- test+hpo))"

results = pd.concat([og, msr, og_more_test_samples_res, more_test_samples_res])
reports = pd.concat([og_creport, more_samples_cr, og_more_test_cr, more_test_cr])

results.to_csv("thesisgan/output/ctabgan_best_model_2/ctabgan_best_results_2.csv", index=False)
reports.to_csv("thesisgan/output/ctabgan_best_model_2/ctabgan_best_reports_2.csv", index=False)
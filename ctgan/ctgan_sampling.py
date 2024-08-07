import pandas as pd
import numpy as np
import pickle
import json
import sys
sys.path.append('.')

from ctabganplus.model.evaluation import get_utility_metrics
from sklearn.preprocessing import LabelEncoder

# set seed 
np.random.seed(42)

ctabgan_best = pickle.load(open('thesisgan/output/ctgan_best_model/pklmodel.pkl', 'rb'))
train_data = pd.read_csv('thesisgan/input/new_train_data.csv')
hpo_data = pd.read_csv("thesisgan/input/new_hpo_data.csv")
test_data = pd.read_csv("thesisgan/input/new_test_data.csv")
og_syn_data = pd.read_csv("thesisgan/output/ctgan_best_model/syn.csv")

more_samples = ctabgan_best.sample(train_data.shape[0])
    
le_dict = {"attack_type": "le_attack_type", "label": "le_label", "proto": "le_proto", "tos": "le_tos"}
for c in le_dict.keys():
    le_dict[c] = LabelEncoder()
    test_data[c] = le_dict[c].fit_transform(test_data[c])
    train_data[c] = le_dict[c].fit_transform(train_data[c])
    hpo_data[c] = le_dict[c].fit_transform(hpo_data[c])
    og_syn_data[c] = le_dict[c].fit_transform(og_syn_data[c])
    more_samples[c] = le_dict[c].fit_transform(more_samples[c])

og_cresults, og_creport = get_utility_metrics(train_data, hpo_data, og_syn_data, scaler="MinMax", type={"Classification":["xgb","lr","dt","rf","mlp"]})
og = og_cresults.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
og["Type"] = og.index
og["Desc"] = "Original with hpo sized samples (train-hpo (hposized))"
og_creport["Desc"] = "Original with hpo sized samples (train-hpo (hposized))"

more_samples_results, more_samples_cr = get_utility_metrics(train_data, hpo_data, more_samples, scaler="MinMax",type={"Classification":["xgb","lr","dt","rf","mlp"]})
msr = more_samples_results.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
msr["Desc"] = "Original with train sized samples (train-hpo (trainsized))"
msr["Type"] = msr.index
more_samples_cr["Desc"] = "Original with train sized samples (train-hpo (trainsized))"

test_data = pd.concat([test_data, hpo_data])
test_data.reset_index(drop=True, inplace=True)

og_more_test_samples, og_more_test_cr = get_utility_metrics(train_data, test_data, og_syn_data, scaler="MinMax",type={"Classification":["xgb","lr","dt","rf","mlp"]})
og_more_test_samples_res = og_more_test_samples.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
og_more_test_samples_res["Desc"] = "Original with hpo sized samples, more test data (train- test+hpo))"
og_more_test_samples_res["Type"] = og_more_test_samples_res.index
og_more_test_cr["Desc"] = "Original with hpo sized samples, more test data (train- test+hpo))"

more_test_samples, more_test_cr = get_utility_metrics(train_data, test_data, more_samples, scaler="MinMax",type={"Classification":["xgb","lr","dt","rf","mlp"]})
more_test_samples_res = more_test_samples.drop(["Model"],axis=1).groupby(["Type"]).mean().sort_values(by="F1_Score", ascending=False).head(100)
more_test_samples_res["Desc"] = "Original with train sized samples, more test data (train- test+hpo))"
more_test_samples_res["Type"] = more_test_samples_res.index
more_test_cr["Desc"] = "Original with train sized samples, more test data (train- test+hpo))"

results = pd.concat([og, msr, og_more_test_samples_res, more_test_samples_res])
reports = pd.concat([og_creport, more_samples_cr, og_more_test_cr, more_test_cr])

results.to_csv("thesisgan/output/ctgan_best_model/table/ctgan_best_results.csv", index=False)
reports.to_csv("thesisgan/output/ctgan_best_model/table/ctgan_best_reports.csv", index=False)
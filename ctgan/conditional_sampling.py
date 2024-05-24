# for conditional sampling - the 2 underrepresented classes in the original sampling are already oversampled
# for the binary classification, the results are already good so we don't need to do conditional sampling
# now we try on the multiclass classification - equally distribute the samples

import pandas as pd
import numpy as np
import sys
sys.path.append(".")
from ctabganplus.model.evaluation import get_utility_metrics

train_data = pd.read_csv("thesisgan/input/new_train_data.csv")
test_data = pd.read_csv("thesisgan/input/new_hpo_data.csv")
equal_dist_multi = pd.read_csv("thesisgan/input/ctgan_equally_distributed_data.csv")
equal_dist_binary = pd.read_csv("thesisgan/input/ctgan_equally_distributed_labels.csv")

attack_type_le = {"benign": 0, "bruteForce": 1, "portScan": 2, "pingScan": 3, "dos": 4}
proto_le = {"TCP": 0, "UDP": 1, "ICMP": 2, "IGMP": 3}
label_type_le = {"normal": 0, "attack": 1, "attacker": 1, "victim": 1}
tos_le = {0 : 0, 32 : 1, 192 : 2, 16 : 3}

#based on the unique values in the dataset, we will create a dictionary to map the values to integers
datasets = [train_data, test_data, equal_dist_multi, equal_dist_binary]
for dataset in datasets:
    dataset["attack_type"] = dataset["attack_type"].map(attack_type_le)
    dataset["proto"] = dataset["proto"].map(proto_le)
    dataset["tos"] = dataset["tos"].map(tos_le)
    dataset["label"] = dataset["label"].map(label_type_le)

print("Training multi-class models with equally distributed data")
equal_dist_multi_results, equal_dist_multi_cr = get_utility_metrics(train_data, test_data, 
                                                                    equal_dist_multi, scaler="MinMax",
                                                                    type={"Classification":["xgb","lr","dt","rf","mlp"]})

print("Training binary models with equally distributed data")
equal_dist_binary_results, equal_dist_binary_cr = get_utility_metrics(train_data, test_data,
                                                                        equal_dist_binary, scaler="MinMax",
                                                                        type={"Classification":["xgb","lr","dt","rf","mlp"]},
                                                                        binary=True)

equal_dist_multi_results.to_csv("thesis-project/thesisgan/output/ctgan_conditional/equal_dist_multi_results.csv", index=False)
equal_dist_multi_cr.to_csv("thesis-project/thesisgan/output/ctgan_conditional/equal_dist_multi_cr.csv", index=False)
equal_dist_binary_results.to_csv("thesis-project/thesisgan/output/ctgan_conditional/equal_dist_binary_results.csv", index=False)
equal_dist_binary_cr.to_csv("thesis-project/thesisgan/output/ctgan_conditional/equal_dist_binary_cr.csv", index=False)




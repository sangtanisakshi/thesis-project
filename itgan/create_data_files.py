import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse

# Write a function that takes a DataFrame and makes a list of the following form:
# [{"name": "label", "type": "categorical","i2s": ["suspicious", "normal", "unknown", "attacker", "victim"], "size": 5,
# {"max": 0.9960784314, "min": 0.0, "name": "src_ip_1", "type": "continuous"}]
# where the first one is a categorical variable and the second one is a continuous variable.
def get_metadata(df: pd.DataFrame) -> list:
    """ Get metadata of the dataset

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        list: Metadata of the dataset
    """
    metadata = []
    for column in df.columns:
        if df[column].dtype in ('object','int64'):
            metadata.append({"name": column, "type": "categorical", "i2s": df[column].astype(str).unique().tolist(), "size": len(df[column].unique())})
        else:
            metadata.append({"max": df[column].max(), "min": df[column].min(), "name": column, "type": "continuous"})
    return metadata

# Write a function that takes the metadata from the function before and create a json file with two entries "columns" with the metadata from before
# and "problem_type" that just takes an input string. Then save the file to a specified path.
def write_metadata(metadata: list, problem_type: str, path: Path) -> None:
    """ Write metadata to a json file

    Args:
        metadata (list): Metadata of the dataset
        problem_type (str): Type of problem
        path (Path): Path to save the file
    """
    metadata_dict = {"columns": metadata, "problem_type": problem_type}
    with open(path, 'w') as file:
        json.dump(metadata_dict, file)

# Write a function that takes a DataFrame and creates an npz file with the following format:
# - two keys: "train" and "test", based on some user defined split ratio
# - each key is a numpy array with the same number of columns as the input DataFrame
# - each categorical column is encoded as integers according to the metadata
def write_npz(train: pd.DataFrame, test: pd.DataFrame, metadata: list, path: Path) -> None:
    """ Write DataFrame to a npz file

    Args:
        df (pd.DataFrame): DataFrame
        metadata (list): Metadata of the dataset
        path (Path): Path to save the file
        split_ratio (float, optional): Split ratio. Defaults to 0.8.
    """
    for column in metadata:
        if column["type"] == "categorical":
            i2s = column["i2s"]
            train[column["name"]] = train[column["name"]].astype(str).apply(lambda x: i2s.index(x))
            test[column["name"]] = test[column["name"]].astype(str).apply(lambda x: i2s.index(x))
    np.savez(path, train=train.to_numpy(), test=test.to_numpy())
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="malware_binary")
    argparser.add_argument("--problem_type", type=str, default="binary_classification")
    argparser.add_argument("--train_path", type=str, default="thesisgan/input/new_train_data.csv")
    argparser.add_argument("--test_path", type=str, default="thesisgan/input/new_hpo_data.csv")
    argparser.add_argument("--output_path", type=str, default="thesisgan/input/")
    argparser.add_argument("--seed", type=int, default=42)
    args = argparser.parse_args()
    np.random.seed(args.seed)
    save_path_npz_train = Path(args.output_path) / Path(args.dataset + ".npz")
    save_path_json_train = Path(args.output_path) / Path(args.dataset + ".json")
    save_path_json_test = Path(args.output_path) / Path(args.dataset + "_test.json")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    cols = list(train_df.columns)
    cols.remove('label')
    cols.append('label')
    train_df = train_df[cols]
    test_df = test_df[cols]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    attack_type_le = {"benign": 0, "bruteForce": 1, "portScan": 2, "pingScan": 3, "dos": 4}
    proto_le = {"TCP": 0, "UDP": 1, "ICMP": 2, "IGMP": 3}
    #label_type_le = {"normal": 0, "attacker": 1, "victim": 2} -- multiclass
    label_type_le = {"normal": 0, "attacker": 1, "victim": 1, "attack": 1} # -- binary
    tos_le = {0 : 0, 32 : 1, 192 : 2, 16 : 3}

    #based on the unique values in the dataset, we will create a dictionary to map the values to integers
    datasets = [train_df, test_df]
    for dataset in datasets:
        dataset["attack_type"] = dataset["attack_type"].map(attack_type_le)
        dataset["proto"] = dataset["proto"].map(proto_le)
        dataset["tos"] = dataset["tos"].map(tos_le)
        dataset["label"] = dataset["label"].map(label_type_le)
    
    train_metadata = get_metadata(train_df)
    test_metadata = get_metadata(test_df)
    write_metadata(train_metadata, args.problem_type, save_path_json_train)
    write_metadata(test_metadata, args.problem_type, save_path_json_test)
    write_npz(train_df, test_df, train_metadata, save_path_npz_train)
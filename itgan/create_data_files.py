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
def write_npz(df: pd.DataFrame, metadata: list, path: Path, split_ratio: float = 0.8) -> None:
    """ Write DataFrame to a npz file

    Args:
        df (pd.DataFrame): DataFrame
        metadata (list): Metadata of the dataset
        path (Path): Path to save the file
        split_ratio (float, optional): Split ratio. Defaults to 0.8.
    """
    train = df.sample(frac=split_ratio, random_state=42)
    test = df.drop(train.index)
    for column in metadata:
        if column["type"] == "categorical":
            i2s = column["i2s"]
            train[column["name"]] = train[column["name"]].astype(str).apply(lambda x: i2s.index(x))
            test[column["name"]] = test[column["name"]].astype(str).apply(lambda x: i2s.index(x))
    np.savez(path, train=train.to_numpy(), test=test.to_numpy())
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="malware")
    argparser.add_argument("--problem_type", type=str, default="multiclass_classification")
    argparser.add_argument("--data_path", type=str, default="../thesisGAN/input/trainval_data.csv")
    argparser.add_argument("--output_path", type=str, default="util/data")
    argparser.add_argument("-sr", "--split_ratio", type=float, default=0.77951)
    argparser.add_argument("--seed", type=int, default=42)
    args = argparser.parse_args()
    np.random.seed(args.seed)
    read_path = Path(args.data_path)
    save_path_npz = Path(args.output_path) / Path(args.dataset + ".npz")
    save_path_json = Path(args.output_path) / Path(args.dataset + ".json")
    df = pd.read_csv(read_path)
    metadata = get_metadata(df)
    write_metadata(metadata, args.problem_type, save_path_json)
    write_npz(df, metadata, save_path_npz, args.split_ratio)
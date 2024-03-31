
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

print(torch.cuda.get_device_name(0))

def remove_million(val):
    """
    Removes the 'M' or 'K' suffix from a string and converts it to a numeric value.

    Args:
        val (str or int): The value to be processed. If it is a string, it should contain a suffix 'M' or 'K'.

    Returns:
        str or int: The processed value. If the input is a string, it is converted to an integer.

    Example:
        >>> remove_million('1.5 M')
        1500000
        >>> remove_million('2.3 K')
        2300
        >>> remove_million(100)
        100
    """
    if type(val) is str:                
        val = val.strip()
        if ' M' in val:                    
            val = val.replace('.', '')
            val = val.replace(' M', '00000')  
        if ' K' in val:                    
            val = val.replace('.', '')
            val = val.replace(' K', '000')                
        return val
    elif type(val) is int:
        return val
    
def ip_process(x: pd.DataFrame, ip : str):
    ip_1 = []
    ip_2 = []
    ip_3 = []
    ip_4 = []
    mod_dict = {'10200':4, '10455':6, '10710':7, '10965':8, '11475':9, '11730':10, '12495':12, '13005':13,
                '13515':14, '13770':16, '14025':17, '14280':18, '14535':19, '14790':20, '15045':21, '15300':22,
                '15555':23, '15810':24, '16065':25, '16320':26, '16575':27, '16830':28, '17085':29, '17340':31,
                '17595':32, '17850':33}
    c = "src_ip_addr" if ip == "src_" else "dst_ip_addr"
    for value in x[c]:
        if "." in str(value):
            a = value.split(".", 4)
            ip_1.append(a[0])
            ip_2.append(a[1])
            ip_3.append(a[2])
            ip_4.append(a[3])
        elif "DNS" in str(value):
            b = np.array([3, 3, 3, 3])
            ip_1.append(b[0])
            ip_2.append(b[1])
            ip_3.append(b[2])
            ip_4.append(b[3])
        elif "_" in str(value):
            p = value.split("_", 2)
            if "OPENSTACK" in p[0]:
                b = np.array([2, 2, 2, 2])
            elif "EXT" in p[0]:
                b = np.array([1, 1, 1, 1])
            else:
                if (int(p[0]) % 255)==0:
                    b = np.array([mod_dict[p[0]], mod_dict[p[0]], mod_dict[p[0]], int(p[1])])
                else:
                    b = np.array([(int(p[0]) % 255), (int(p[0]) % 255), (int(p[0]) % 255), int(p[1])])
            ip_1.append(b[0])
            ip_2.append(b[1])
            ip_3.append(b[2])
            ip_4.append(b[3])
    
    ip_data = pd.DataFrame({(ip+'ip_1'): ip_1, (ip+'ip_2'): ip_2, (ip+'ip_3'): ip_3, (ip+'ip_4'): ip_4},dtype=np.float64)
    return ip_data

def hex_to_tcp_flags(hex_value):
    """
    Converts a hexadecimal value to TCP flags.

    Args:
        hex_value (str): The hexadecimal value to convert.

    Returns:
        str: The TCP flags represented as a string.

    Example:
        >>> hex_to_tcp_flags('0x52')
        '.E.A..S.'
    """
    binary_value = bin(int(hex_value, 16))[2:].zfill(8)
    flags = ['C', 'E', 'U', 'A', 'P', 'R', 'S', 'F']
    tcp_flags = ''.join([flags[i] if bit == '1' else '.' for i, bit in enumerate(binary_value)])
    return tcp_flags

def main(file_path, model):

    external_csv = []
    openstack_csv = []
    import os
    for dirname, _, filenames in os.walk(file_path):
        for filename in filenames:
            if 'external' in filename:
                fx = os.path.join(dirname, filename)
                external_csv.append(fx)            
                print(fx)
            elif 'internal' in filename:
                fx = os.path.join(dirname, filename)
                openstack_csv.append(fx)
                print(fx)
                
    csvs = external_csv + openstack_csv
    df = pd.concat(objs=[pd.read_csv(fp, encoding='utf-8') for fp in csvs], ignore_index=True, copy=False)
    
    # Rename columns
    df = df.rename(str.lower, axis='columns')
    df = df.rename(str.strip, axis='columns')
    df.rename(columns={
                'date first seen': 'date_first_seen', 
                'src ip addr': 'src_ip_addr',
                'src pt': 'src_pt',
                'dst ip addr': 'dst_ip_addr',
                'dst pt': 'dst_pt',
                'attacktype': 'attack_type',
                'attackid': 'attack_id',
                'attackdescription': 'attack_description',
                'class': 'label'
            }, inplace=True)

    df = df.loc[(df["date_first_seen"]>="2017-03-17 14:18:05") & (df["date_first_seen"]<="2017-03-20 17:42:17")]
    df.reset_index(drop=True, inplace=True)
    
    # Remove million suffix from 'bytes' column
    df['bytes'] = df['bytes'].apply(remove_million)
    df['bytes'] = pd.to_numeric(df['bytes'], errors='raise', downcast='float')
    
    # Remove NA and INF values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("N/A rows after preproc", df.isna().any(axis=1).sum())
    df.dropna(inplace=True)
    
    # Convert hexadecimal TCP flags to custom format
    df["flags"] = df["flags"].apply(lambda x:hex_to_tcp_flags(x) if '0x' in x else x)
    df['flags'] = df['flags'].apply(lambda x: '.' * (8 - len(str(x))) + str(x))

    # One-hot encode TCP flags
    df['flags'] = df['flags'].str.strip()
    data_df = df['flags'].apply(func=lambda flag_str: [0 if c == '.' else 1 for c in flag_str]).to_list()
    columns_df = ['tcp_con','tcp_ech','tcp_urg', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin']
    ohe_flag_data = pd.DataFrame(data=data_df,columns=columns_df,dtype=int)
    flags_idx = df.columns.get_loc('flags')
    for i, c in enumerate(ohe_flag_data.columns):
        df.insert(loc=flags_idx+i, column=c, value=ohe_flag_data[c])

    # Preprocess other columns
    df["proto"] = df["proto"].str.strip()
    df['attack_type'] = df['attack_type'].replace({'---': 'benign'})
    df['attack_id'] = df['attack_id'].replace({'---': 0})
    df['attack_id'] = df['attack_id'].astype(np.int32)
    df = df[~df['proto'].isin(['IGMP', 'GRE'])]
    df.reset_index(drop=True, inplace=True)
    
    df["date_first_seen"] = pd.to_datetime(df["date_first_seen"], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df['day_of_week'] = df['date_first_seen'].dt.day_name()

    ## convert date_first_seen to total seconds
    df['date_first_seen'] = df['date_first_seen'].apply(lambda x: x.time())
    df['date_first_seen'] = df['date_first_seen'].apply(lambda x: (x.hour * 3600) + (x.minute * 60) + (x.second))
    
    # convert date first seen to time of day in seconds with 3 decimal places
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    df['date_first_seen'] = df['date_first_seen']/86400
    df.rename(columns={'date_first_seen':'time_of_day'}, inplace=True)
    
    #normalize values
    df["src_pt"] = df["src_pt"]/df["src_pt"].max()
    df["dst_pt"] = df["dst_pt"]/df["dst_pt"].max()
    columns_df = ['duration', 'bytes', 'packets']
    for c in columns_df:
        if c == "duration":
            df[c] = np.log(df[c]+1)
        else:
            df[c] = np.log(df[c])
        df[c] = (df[c]-df[c].min())/(df[c].max()-df[c].min())
    
    # convert ip address to continuous values
    ip_data_src = ip_process(df, "src_")
    ip_data_dst = ip_process(df, "dst_")
    
    df = pd.concat([df, ip_data_src, ip_data_dst], axis=1)
    
    #normalize the ip values
    for c in ['src_ip_1', 'src_ip_2', 'src_ip_3', 'src_ip_4', 'dst_ip_1', 'dst_ip_2', 'dst_ip_3', 'dst_ip_4']:
        df[c] = df[c].astype(np.float64)
        df[c] = df[c]/255
    
    # if all 4 different -> normal ip
    # if 0 0 0 0 -> normal ip
    # if 255, 255, 255, 255 -> normal ip
    # if ext -> 1 1 1 1
    # if openstack -> 2 2 2 2
    # if dns -> 3 3 3 3
    # if none of these -
    # then 
    # if first 3 same (not 1,2,3,0,255) then its the anonymized ip
    

    # drop columns that are not needed
    df.drop(columns=['src_ip_addr', 'dst_ip_addr','attack_description','flows','flags'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # remove duplicated rows
    dupli = df.duplicated().sum()
    if dupli > 0:
        print(dupli, "fully duplicate rows")
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    print("Total data after preprocessing:", (df.shape))
    print("NA rows check:", df.isna().sum())
    df.to_csv('thesisGAN/input/preprocessed.csv')
    print("Preprocessing done. Subset data saved to input folder")

    # Split the data into train test and val
    train_data = df.sample(frac=0.7, random_state=42)
    print("Total training data:", (train_data.shape))
    val_data = df.drop(train_data.index).sample(frac=0.66, random_state=42)
    print("Total validation data:", (val_data.shape))
    test_data = df.drop(train_data.index).drop(val_data.index)
    print("Total test data:", (test_data.shape))
    trainval = train_data._append(val_data)
    
    train_data.to_csv('thesisGAN/input/train_data.csv', index=False)
    test_data.to_csv('thesisGAN/input/test_data.csv', index=False)
    val_data.to_csv('thesisGAN/input/val_data.csv', index=False)
    trainval.to_csv('thesisGAN/input/trainval_data.csv', index=False)
    
    if model == "ITGAN":
        print("Preprocessing done. Data saved to input folder. Please run the create_data_files.py script to create the npz and metadata files for running ITGAN")
    elif model == "CTGAN":
        print("Preprocessing done for CTGAN. Data saved to input folder")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-fp','--file_path', type=str, help='Path to the csv file to be preprocessed')
    parser.add_argument('-m','--model', type=str, help='Model for which the data is being preprocessed')
    args = parser.parse_args()
    
    main(args.file_path, args.model)
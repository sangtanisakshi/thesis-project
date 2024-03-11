
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

# def remove_million(val):
#     if type(val) is str:                
#         val = val.strip()
#         if ' M' in val:                    
#             val = val.replace('.', '')
#             val = val.replace(' M', '00000')  
#         if ' K' in val:                    
#             val = val.replace('.', '')
#             val = val.replace(' K', '000')                
#         return val
#     elif type(val) is int:
#         return val

# def hex_to_tcp_flags(hex_value):
#     binary_value = bin(int(hex_value,16))[2:].zfill(8)
#     flags = ['C','E','U', 'A', 'P', 'R', 'S', 'F']
#     tcp_flags = ''.join([flags[i] if bit == '1' else '.' for i, bit in enumerate(binary_value)])
#     return tcp_flags

# def main(file_path):

#     # Input data files are available in the read-only "../input/" directory
#     # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#     external_csv = []
#     openstack_csv = []
#     import os
#     for dirname, _, filenames in os.walk(file_path):
#         for filename in filenames:
#             if 'external' in filename:
#                 fx = os.path.join(dirname, filename)
#                 external_csv.append(fx)            
#                 print(fx)
#             elif 'internal' in filename:
#                 fx = os.path.join(dirname, filename)
#                 openstack_csv.append(fx)
#                 print(fx)
                
#     csvs = external_csv + openstack_csv
#     df = pd.concat(objs=[pd.read_csv(fp, encoding='utf-8') for fp in csvs], ignore_index=True, copy=False)
    
#     # Rename columns
#     df = df.rename(str.lower, axis='columns')
#     df = df.rename(str.strip, axis='columns')
#     df.rename(columns={
#                 'date first seen': 'date_first_seen', 
#                 'src ip addr': 'src_ip_addr',
#                 'src pt': 'src_pt',
#                 'dst ip addr': 'dst_ip_addr',
#                 'dst pt': 'dst_pt',
#                 'attacktype': 'attack_type',
#                 'attackid': 'attack_id',
#                 'attackdescription': 'attack_description',
#                 'class': 'label'
#             }, inplace=True)

#     # Remove million suffix from 'bytes' column
#     df['bytes'] = df['bytes'].apply(remove_million)
#     df['bytes'] = pd.to_numeric(df['bytes'], errors='raise', downcast='float')
    
#     # Remove NA and INF values
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     print("N/A rows after preproc", df.isna().any(axis=1).sum())
#     df.dropna(inplace=True)
    
#     # Convert hexadecimal TCP flags to custom format
#     df["flags"] = df["flags"].apply(lambda x:hex_to_tcp_flags(x) if '0x' in x else x)
#     df['flags'] = df['flags'].apply(lambda x: '.' * (8 - len(str(x))) + str(x))

#     # One-hot encode TCP flags
#     df['flags'] = df['flags'].str.strip()
#     data_df = df['flags'].apply(func=lambda flag_str: [0 if c == '.' else 1 for c in flag_str]).to_list()
#     columns_df = ['tcp_con','tcp_ech','tcp_urg', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin']
#     ohe_flag_data = pd.DataFrame(data=data_df,columns=columns_df,dtype=int)
#     flags_idx = df.columns.get_loc('flags')
#     for i, c in enumerate(ohe_flag_data.columns):
#         df.insert(loc=flags_idx+i, column=c, value=ohe_flag_data[c])

#     # Preprocess other columns
#     df["proto"] = df["proto"].str.strip()
#     df['attack_type'] = df['attack_type'].replace({'---': 'benign'})
#     df['attack_id'] = df['attack_id'].replace({'---': 0})
#     df['attack_id'] = df['attack_id'].astype(np.int32)
    
    
#     # Drop duplicates
#     df.drop_duplicates(inplace=True)  
#     df.reset_index(drop=True, inplace=True)
    
#     # Get a subset of the data as the main dataset
#     df_sub = df.loc[(df["date_first_seen"]>="2017-03-17 14:18:05") & (df["date_first_seen"]<="2017-03-20 17:42:17")]
#     df_sub.reset_index(drop=True, inplace=True)
    
#     # Save the cleaned data to 2 new files
#     df.to_csv('input/clean_data_all.csv')
#     print("Preprocessing done. Cleaned entire data saved to input/cleaned_data_all.csv")
    
#     df_sub.to_csv('input/clean_data_sub.csv')
#     print("Preprocessing done. Subset data saved to input/cleaned_data_sub.csv")
    
#     # Drop columns that are not needed and drop duplicates
#     df_sub.drop(labels=['src_ip_addr', 'src_pt', 'dst_ip_addr', 'dst_pt','attack_description','flows','flags'], axis=1, inplace=True)
#     df_sub.drop_duplicates(inplace=True)
#     df_sub.reset_index(drop=True, inplace=True)

#     # Save the cleaned data to a new file
#     df_sub.to_csv('input/sub_data_temporal.csv')
#     print("Preprocessing done. Temporal subset data saved to inputt/sub_data_temporal.csv")
     
#     # Remove the date column, drop more duplicates and save the file
#     df_sub.drop(labels=['date_first_seen'], axis=1, inplace=True)
#     df_sub.drop_duplicates(inplace=True)
#     df_sub.reset_index(drop=True, inplace=True)
    
#     df_sub.to_csv('input/sub_data_non_temporal.csv')
#     print("Preprocessing done. Non temporal subset data saved to input/sub_data_non_temporal.csv")
    
# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser(description='Preprocess data')
#     parser.add_argument('-fp','--file_path', type=str, help='Path to the csv file to be preprocessed')
#     args = parser.parse_args()
    
#     main(args.file_path)
# from ctgan import CTGAN
# from ctgan import load_demo

# real_data = load_demo()

# # Names of the columns that are discrete
# discrete_columns = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income'
# ]

# ctgan = CTGAN(epochs=10)
# ctgan.fit(real_data, discrete_columns)

# # Create synthetic data
# synthetic_data = ctgan.sample(1000)

# print(synthetic_data.head(10))

from ctgan import CTGAN
import pandas as pd

real_data = pd.read_csv('thesisGAN/input/clean_data_sub.csv')
real_data = real_data.drop(columns=['date_first_seen','flags','attack_description','Unnamed: 0','flows','src_ip_addr','src_pt','dst_ip_addr','dst_pt','attack_id'])
real_data["packets"] = real_data["packets"].astype('float64')

print(real_data.info())

# Names of the columns that are discrete
discrete_columns = [
'proto',
'tcp_con',
'tcp_ech',
'tcp_urg',
'tcp_ack',
'tcp_psh',
'tcp_rst',
'tcp_syn',
'tcp_fin',
'tos',
'label',
'attack_type'
]

ctgan = CTGAN(epochs=10)

ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
print(synthetic_data.head(10))
synthetic_data.to_csv('thesisGAN/model-output/synthetic_data.csv', index=False)
print("Synthetic data saved. Check the output folder.")

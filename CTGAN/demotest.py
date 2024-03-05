from ctgan import CTGAN
import pandas as pd

real_data = pd.read_csv('../thesisGAN/input/clean_data_sub.csv')
real_data = real_data.drop(columns=['attack_description','flags','attack_description','Unnamed: 0'])
real_data["packets"] = real_data["packets"].astype('float64')
real_data["dst_pt"] = real_data["dst_pt"].astype('int64')

print(real_data.info())

# Names of the columns that are discrete
discrete_columns = [
'date_first_seen',
'proto',
'src_ip_addr',
'src_pt',
'dst_ip_addr',
'dst_pt',
'flows',
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
'attack_type',
'attack_id'
]

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
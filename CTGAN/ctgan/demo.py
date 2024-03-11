from ctgan import CTGAN
import pandas as pd
import torch

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

print(torch.cuda.get_device_name(0))

real_data = pd.read_csv('thesisGAN/input/preprocessed.csv')

# Split the data into train test and val
train_data = real_data.sample(frac=0.7, random_state=42)
test_data = real_data.drop(train_data.index).sample(frac=0.66, random_state=42)
val_data = real_data.drop(train_data.index).drop(test_data.index)  

train_data.to_csv('thesisGAN/input/train_data.csv', index=False)
test_data.to_csv('thesisGAN/input/test_data.csv', index=False)
val_data.to_csv('thesisGAN/input/val_data.csv', index=False)

#Names of the columns that are discrete
discrete_columns = [
'label',
'attack_type',
'attack_id',
'proto',
'day_of_week',
'tcp_con',
'tcp_ech',
'tcp_urg',
'tcp_ack',
'tcp_psh',
'tcp_rst',
'tcp_syn',
'tcp_fin']

ctgan = CTGAN(epochs=1)

ctgan.fit(train_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
print(synthetic_data.head(20))
synthetic_data.to_csv('thesisGAN/model-outputs/synthetic_data.csv', index=False)
print("Synthetic data saved. Check the output folder.")

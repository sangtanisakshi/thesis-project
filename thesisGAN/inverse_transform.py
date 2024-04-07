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

    # if all 4 different -> normal ip
    # if 0 0 0 0 -> normal ip
    # if 255, 255, 255, 255 -> normal ip
    # if ext -> 1 1 1 1
    # if openstack -> 2 2 2 2
    # if dns -> 3 3 3 3
    # if none of these -
    # then 
    # if first 3 same (not 1,2,3,0,255) then its the anonymized ip

def reverse_ip_process(ip_data: pd.DataFrame, ip: str):
    ip_1 = ip_data[ip + 'ip_1']
    ip_2 = ip_data[ip + 'ip_2']
    ip_3 = ip_data[ip + 'ip_3']
    ip_4 = ip_data[ip + 'ip_4']
    
    mod_dict = {4: '10200', 6: '10455', 7: '10710', 8: '10965', 9: '11475', 10: '11730', 12: '12495', 13: '13005',
                14: '13515', 16: '13770', 17: '14025', 18: '14280', 19: '14535', 20: '14790', 21: '15045', 22: '15300',
                23: '15555', 24: '15810', 25: '16065', 26: '16320', 27: '16575', 28: '16830', 29: '17085', 31: '17340',
                32: '17595', 33: '17850'}
    
    c = ip + 'ip_addr'
    #unnormalize the ip
    ip_1 = ip_1 * 255
    ip_2 = ip_2 * 255
    ip_3 = ip_3 * 255
    ip_4 = ip_4 * 255
    values = []
    for i in range(len(ip_1)):
        if ip_1[i] == 3 and ip_2[i] == 3 and ip_3[i] == 3 and ip_4[i] == 3:
            values.append('DNS')
        elif ip_1[i] == 2 and ip_2[i] == 2 and ip_3[i] == 2 and ip_4[i] == 2:
            values.append('OPENSTACK')
        elif ip_1[i] == 1 and ip_2[i] == 1 and ip_3[i] == 1 and ip_4[i] == 1:
            values.append('EXT')
        else:
            # if first three same and last different then it is anonymized, check mod_dict
            if ip_1[i] == ip_2[i] and ip_2[i] == ip_3[i] and ip_3[i] != ip_4[i]:
            
                        else:
                value = str(ip_1[i]) + '_' + str(ip_4[i])
            values.append(value)
    
    ip_data[c] = values
    return ip_data

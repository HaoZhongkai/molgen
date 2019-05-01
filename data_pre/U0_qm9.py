from config import config
import pickle
import rdkit.Chem as Chem
from Funcs import smiles2data
import torch
import random
import numpy as np

dconfig = config()
smi_path = '/home/jeffzhu/nips_gail/Dataset/dataset/a_smiles.npy'
pro_path = '/home/jeffzhu/nips_gail/Dataset/dataset/a_property.npy'

def sample_U0(smiles,properties,max_num,config):
    sample_arr = random.sample(range(len(smiles)),max_num)
    U0_list = []
    smile_list = []
    iter = 0
    for i in sample_arr:
        U0_list.append(1e-3*properties[i,10])
        smile_list.append(smiles[i])
        print(iter)
        iter = iter + 1

    node_arr,adj = smiles2data(smile_list,config.max_atom_num,config.possible_atoms,config.possible_bonds,True)
    U0 = np.array(U0_list)
    adj = torch.Tensor(adj)
    node_arr = torch.Tensor(node_arr)
    U0 = torch.Tensor(U0)
    return node_arr,adj,U0

smiles = list(np.load(smi_path))
pro = np.load(pro_path)

train = sample_U0(smiles,pro,80000,dconfig)
valid = sample_U0(smiles,pro,8000,dconfig)
test = sample_U0(smiles,pro,8000,dconfig)

with open(dconfig.DATASET_PATH+'/gdb9_U0.pkl','wb') as fp:
    pickle.dump((train,valid,test),fp)
print('OK')
'''数据存储格式:(train,valid,test)'''

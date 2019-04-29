'''采样一些分子,按qed值存储'''
from data_pre.sparse_molecular_dataset import SparseMolecularDataset
from config import config
import pickle
import rdkit.Chem as Chem
import rdkit.Chem.QED as QED
import random
import torch
from Funcs import smiles2data
import numpy as np
dconfig = config()
data = SparseMolecularDataset()
data.load(dconfig.DATASET_PATH+'/gdb9_9nodes.sparsedataset')
# data = list(np.load('/home/jeffzhu/MCTs/dataset/datasets/smiles_data.npy'))
def sample_qed(data_smiles,max_num):
    exam_molecules_smiles = random.sample(data_smiles,max_num)
    qed_list = []
    for i in range(len(exam_molecules_smiles)):
        qed_list.append(QED.qed(Chem.MolFromSmiles(exam_molecules_smiles[i])))
        print(i,exam_molecules_smiles[i],qed_list[i])
    return (exam_molecules_smiles,qed_list)

def _to_tensor(data,config):
    smiles = data[0]
    datas = data[1]
    node_arr,adj = smiles2data(smiles,config.max_atom_num,config.possible_atoms,config.possible_bonds,batch=True)


    adj = torch.Tensor(adj)
    node_arr = torch.Tensor(node_arr)
    datas = torch.Tensor(datas)
    return node_arr,adj,datas



data_smiles = list(data.smiles)
# data_smiles = data
train = sample_qed(data_smiles,120000)
valid = sample_qed(data_smiles,8000)
test = sample_qed(data_smiles,8000)

train = _to_tensor(train,config=dconfig)
valid = _to_tensor(valid,config=dconfig)
test = _to_tensor(test,config=dconfig)


with open(dconfig.DATASET_PATH+'/gdb9_qed.pkl','wb') as fp:
    pickle.dump((train,valid,test),fp)
print('OK')
'''数据存储格式:(train,valid,test)'''
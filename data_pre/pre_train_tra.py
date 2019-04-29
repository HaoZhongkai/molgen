import networkx as nx
from rdkit import Chem
import numpy as np
from data_pre import SparseMolecularDataset
from rdkit.Chem import Draw
from Funcs import *
from config import Config
from .Multi_trajectory import Multi_trajectory
import random
import pickle

'''生成预训练的(r,a) pair,输入data为数据集,从中随机截取max_num个分子,每个分子per_molecule轨迹'''
def generate_pretrain_tra(tra,data_smiles,max_num,max_tra_per_mole = 1):

    trajectories = []
    mol_list = random.sample(data_smiles, max_num)
    for i in mol_list:
        new_tra,_ = tra.sample(mol_list[i])
        trajectories = trajectories + new_tra
        print(i+1)

    return trajectories


default_config = Config()
dataset_path = 'E:\\0Lab\ML\\repo\Model_code\Dataset\gdb9_9nodes.sparsedataset'

dataset = SparseMolecularDataset()
dataset.load(dataset_path)

max_num = 5000
tra = Multi_trajectory(default_config)

traces = generate_pretrain_tra(tra,dataset['smiles'],max_num)

with open(default_config.pre_train_tra_save_path,'w') as fp:
    pickle.dump(fp,traces)

print('sample success')
print(len(traces))







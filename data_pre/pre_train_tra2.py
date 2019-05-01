import sys
sys.path.append('/home/jeffzhu/nips_gail')
import networkx as nx
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from Funcs import *
from config import Config
from data_pre.Multi_trajectory import Multi_trajectory
import random
import pickle

'''生成预训练的(r,a) pair,输入data为数据集,从中随机截取max_num个分子,每个分子per_molecule轨迹'''
def generate_pretrain_tra(tra,data_smiles,max_num,max_tra_per_mole = 1):

    trajectories = []
    mol_list = random.sample(data_smiles, max_num)
    error_num = 0
    for i in range(len(mol_list)):
        if Chem.MolFromSmiles(mol_list[i]).GetNumAtoms()>=7:
            try:
                new_tra,_ = tra.sample(Chem.MolFromSmiles(str(mol_list[i])))
            except Exception:
                error_num = error_num + 1
                print('error:',error_num,'smiles:',mol_list[i])
                continue
            trajectories = trajectories + new_tra
            print(i+1,str(mol_list[i]))

    return trajectories,error_num


default_config = Config()
dataset_path = '/home/jeffzhu/MCTs/dataset/datasets/a_smiles.npy'

dataset = list(np.load(dataset_path))

max_num = 40000
tra = Multi_trajectory(default_config)

traces,errors = generate_pretrain_tra(tra,dataset,max_num)
# traces = []
with open(default_config.random_tra_save_path+'/rand_sample1.pkl','wb') as fp:
    pickle.dump(traces,fp)

print('sample success')

print('traces num:',len(traces))
print('error num:',errors)
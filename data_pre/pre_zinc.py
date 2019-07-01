import networkx as nx
from Funcs import *
from process_data import DataUnit
import pickle
from config import Config
import torch
import numpy as np

table_of_atom = {
    'C': 0,
    'N': 1,
    'O': 2,
    'F': 3,
    'S': 4,
    'Cl': 5,
    'Br': 6,
    'I': 7
}

opt = Config()
zinc_path = '/home/jeffzhu/MCTs/dataset/datasets/zinc_clean.pkl'
zinc_qed_path = '/home/jeffzhu/MCTs/dataset/datasets/zinc_clean_qed_score.pkl'
zinc_J_score_path = '/home/jeffzhu/MCTs/dataset/datasets/zinc_clean_J_score.pkl'

data = pickle.load(open(zinc_path, 'rb'))
qed = pickle.load(open(zinc_qed_path, 'rb'))
J_s = pickle.load(open(zinc_J_score_path, 'rb'))

length_data = len(data)
node_arr = np.zeros([length_data, 38])

for i in range(length_data):
    node_list = data[i].node_list
    for j in range(len(node_list)):
        node_arr[i][j] = table_of_atom[node_list[j]]

adj_pool = []

for i in range(length_data):
    temp_adj = np.zeros([3, 38, 38])
    data_adj = data[i].adj.astype(int)
    length_adj = len(data[i].node_list)

    for j in range(length_adj):
        for k in range(length_adj):
            if data_adj[j][k] != 0:
                temp_adj[data_adj[j][k] - 1][j][k] = 1
    adj_pool.append(temp_adj)

adj = np.stack(adj_pool, axis=0)

# to tensor
node_arr, adj, qeds, J_scores = torch.Tensor(node_arr), torch.Tensor(adj), torch.Tensor(qed), torch.Tensor(J_s)

data = {'node_arr': node_arr, 'adj': adj, 'qed': qeds, 'J_score': J_scores}

pickle.dump(data, open(opt.DATASET_PATH + 'zinc_dataset_clean.pkl', 'wb'))
print('OK')

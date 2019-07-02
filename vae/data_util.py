from torch.utils.data import Dataset
from process_data import DataUnit
import pickle
from torch.utils.data import TensorDataset
import numpy as np



# 取测试集为1000个
# props 为id序列 list of id names
class Zinc_dataset():
    def __init__(self, path, train_num, valid_num, propid, valid=True):
        '''载入数据'''
        self.path = path
        self.valid = True
        self.train_num = train_num
        self.valid_num = valid_num
        self.propid = propid
        self.dataset = {'train': [], 'valid': [], 'test': []}


    def load_data(self):
        all_data = pickle.load(open(self.path, 'rb'))
        adj = all_data['adj']
        node_arr = all_data['node_arr']

        seq = np.random.choice(node_arr.size(0), self.train_num + self.valid_num, replace=False)
        train_seq, test_seq, valid_seq = seq[:int(self.train_num * 0.9)], \
                                         seq[int(self.train_num * 0.9):self.train_num], seq[
                                                                                        self.train_num:self.train_num +
                                                                                                       self.valid_num]
        train_adj, test_adj, valid_adj = adj[train_seq], adj[test_seq], adj[valid_seq]
        train_nodes, test_nodes, valid_nodes = node_arr[train_seq], node_arr[test_seq], node_arr[valid_seq]

        train_props, test_props, valid_props = {}, {}, {}
        for key in self.propid:
            train_props.update({key: all_data[key][train_seq]})
            test_props.update({key: all_data[key][test_seq]})
            valid_props.update({key: all_data[key][valid_seq]})

        # select data 训练集测试集按5:1选取,验证集只取valid_num

        train = (train_nodes, train_adj, train_props)
        test = (test_nodes, test_adj, test_props)
        valid = (valid_nodes, valid_adj, valid_props)

        return train, test, valid




    def Get_data(self):
        train, test, valid = self.load_data()
        train_set = zinc_data(train)
        test_set = zinc_data(test)
        valid_set = zinc_data(valid)

        return train_set, test_set, valid_set




class zinc_data(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = self.data[0].size(0)
        self.keys = self.data[2].keys()
        self.labels = {key: [] for key in self.keys}

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        labels = {key: [] for key in self.keys}
        for key in self.keys:
            labels[key] = self.data[2][key][item]
        return (self.data[0][item], self.data[1][item], labels)

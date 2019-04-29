import torch
from torch.utils.data import dataset
import pickle
import numpy as np
from Funcs import *


class QM9qed_dataset():
    def __init__(self,path,transform=False,valid=True):
        '''载入数据'''
        self.path = path
        self.valid = True



    def Get_data(self):
        train,valid,test = pickle.load(open(self.path,'rb'))
        train_dataset = dataset.TensorDataset(*train)
        test_dataset = dataset.TensorDataset(*test)
        if valid:
            valid_dataset = dataset.TensorDataset(*valid)
            return train_dataset,valid_dataset,test_dataset
        else:
            return train_dataset,test_dataset

class QM9U0_dataset():
    def __init__(self,path,transform=False,valid=True):
        '''载入数据'''
        self.path = path
        self.valid = True



    def Get_data(self):
        train,valid,test = pickle.load(open(self.path,'rb'))
        train_dataset = dataset.TensorDataset(*train)
        test_dataset = dataset.TensorDataset(*test)
        if valid:
            valid_dataset = dataset.TensorDataset(*valid)
            return train_dataset,valid_dataset,test_dataset
        else:
            return train_dataset,test_dataset
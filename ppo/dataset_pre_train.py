from torch.utils.data import Dataset
import pickle
import torch as T
import numpy as np
from Funcs import *
class Tra_Dataset(Dataset):
    def __init__(self,root,max_trace_num,train=True,test=False):
        self.data_all = pickle.load(open(root,'rb'))
        self.max_num = min(len(self.data_all),max_trace_num)

    def __getitem__(self, index):
        N = T.Tensor(self.data_all[index][1]).long()        #数据格式补丁,N加scaffold
        A = T.Tensor(self.data_all[index][0])

        a_first, a_second, a_edge, a_stop = (T.Tensor([np.argmax(self.data_all[index][2][0])]),
                 T.Tensor([np.argmax(self.data_all[index][2][1])]),
                 T.Tensor([np.argmax(self.data_all[index][2][2])]),
                 T.Tensor([self.data_all[index][2][3]]))

        #数据格式补丁
        a_first, a_second, a_edge, a_stop = (a_first.long().squeeze(),a_second.long().squeeze(),a_edge.long().squeeze(),
        a_stop.long().squeeze())

        return N,A,a_first, a_second, a_edge, a_stop

    def __len__(self):
        return self.max_num


import torch.nn as nn
import torch
from .basic_model import Gconv
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GCN_1(nn.Module):
    def __init__(self,config):
        super(GCN_1,self).__init__()
        self.agg_fun = config.agg_fun
        self.embeddim = config.node_feature_dim
        self.act = config.activation

        self.atom_embed = nn.Embedding(len(config.possible_atoms)+1,self.embeddim,padding_idx=0)
        self.gcn_layer1 = Gconv(self.embeddim,self.act)
        self.gcn_layer2 = Gconv(self.embeddim,self.act)
        self.gcn_layer3 = Gconv(self.embeddim,self.act)


    '''输入N为node array,E为邻接矩阵,计算H:node embedding (batch*n*k)'''
    def forward(self,N, E):
        # D:Tensor,通过对E(batch*type*n*n)中对第四维求和再升维得到的batch*type*n*n的张量
        H = self.atom_embed(N)
        # H = torch.randn([N.size(0),9,self.embeddim])
        E = E + torch.eye(E.shape[2])
        D = torch.diag_embed(torch.sum(E, dim=3) ** (-1 / 2))  # 此时D为四维tensor(先通过reduce_sum降维再升维)
        # 通过三个图卷积层
        H = self.gcn_layer1(H,D,E)
        H = self.gcn_layer1(H,D,E)
        H = self.gcn_layer1(H,D,E)
        return H





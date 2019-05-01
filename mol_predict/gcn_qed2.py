from torch import nn
from gcn.basic_model import BasicModel
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
'''这里整个不用AGG函数'''
'''前向传播,H:初始原子的embeddings(Batch*n*k),E:对应不同化学键的邻接tensor(Batch*bond_type*n*n)'''
class Gconv1(nn.Module):
    def __init__(self,embed_dim,act=None,bond_type=4):
        super(Gconv1,self).__init__()
        self.act = act
        self.Weight = Parameter(torch.Tensor(bond_type,embed_dim,embed_dim))
        #parameter initialize
        init.xavier_normal_(self.Weight)

    '''图卷积层,输入预处理之后的图的L矩阵'''
    def forward(self,H,D,E):
        return self.act(torch.einsum('btij,btjk,btkl,btlm,tmd->btid',D,E,D,H,self.Weight))


class GCN_1(nn.Module):
    def __init__(self,config):
        super(GCN_1,self).__init__()
        self.agg_fun = config.agg_fun
        self.embeddim = config.node_feature_dim
        self.act = config.activation
        self.bond_type = 4

        self.atom_embed = nn.Embedding(len(config.possible_atoms)+1,self.embeddim)
        self.gcn_layer1 = Gconv1(self.embeddim,self.act)
        self.gcn_layer2 = Gconv1(self.embeddim,self.act)
        self.gcn_layer3 = Gconv1(self.embeddim,self.act)


    '''输入N为node array,E为邻接矩阵,计算H:node embedding (batch*n*k)'''
    def forward(self,N, E):
        # D:Tensor,通过对E(batch*type*n*n)中对第四维求和再升维得到的batch*type*n*n的张量
        #把H 沿键type维度堆叠
        H = self.atom_embed(N).unsqueeze(1).repeat(1,E.size(1),1,1)
        # H = torch.randn([N.size(0),9,self.embeddim])
        E = E + torch.eye(E.shape[2])
        D = torch.diag_embed(torch.sum(E, dim=3) ** (-1 / 2))  # 此时D为四维tensor(先通过reduce_sum降维再升维)
        # 通过三个图卷积层
        H = self.gcn_layer1(H,D,E)
        H = self.gcn_layer1(H,D,E)
        H = self.gcn_layer1(H,D,E)
        return H

class GCN_QED(BasicModel):
    def __init__(self,config):
        super(GCN_QED,self).__init__(config)
        self.config = config
        self.embeddim = config.node_feature_dim


        self.GCN = GCN_1(config)
        self.MLP = nn.Sequential(
            nn.Linear(self.embeddim,10),
            nn.ReLU(),
            nn.Linear(10,1),
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(config.max_bond_type),
            nn.Conv2d(config.max_bond_type,10,4,2,1),   #feature/2
            nn.ReLU(),
            nn.Conv2d(10,20,4,2,1),   #feature/2
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.1)
        )
        self.MLP = nn.Sequential(
            nn.Linear(int(20*(self.embeddim/4)**2),100),
            nn.ReLU(),
            nn.Linear(100,20),
            nn.ReLU(),
            nn.Linear(20,1)
        )


    def forward(self,N, E):
        H = self.GCN(N, E)


        x = torch.einsum('btji,btjk->btik',[H,H])
        x = self.conv(x)
        x = self.MLP(x.view(x.size(0),-1))
        # x = torch.mean(self.MLP(H),dim=1)

        return x


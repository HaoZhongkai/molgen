from .basic_model import BasicModel
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.distributions import Categorical
from Funcs import *

'''这里整个不用AGG函数'''
'''前向传播,H:初始原子的embeddings(Batch*n*k),E:对应不同化学键的邻接tensor(Batch*bond_type*n*n)'''
class Gconv1(nn.Module):
    def __init__(self,embed_dim,act=None,bond_type=3):
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
        self.bond_type = len(config.possible_bonds)

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
        H = self.gcn_layer2(H,D,E)
        H = self.gcn_layer3(H,D,E)

        return H




class PolicyNet(BasicModel):
    def __init__(self,config):
        super(PolicyNet,self).__init__(config)

        self.config = config
        self.agg_fun = config.agg_fun
        self.act = config.activation
        self.embeddim = config.node_feature_dim
        self.max_dim = len(config.possible_atoms)+config.max_atom_num
        self.max_atom_num = config.max_atom_num
        self.bond_types = len(config.possible_bonds)
        self.GCN = GCN_1(config)

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,10,4,2,1),   #feature/2
            nn.LeakyReLU(0,1),
            nn.Conv2d(10,20,4,2,1),   #feature/2
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.1)
        )

        self.MLP = nn.Sequential(
            nn.Linear(int(20 * (self.embeddim / 4) ** 2), 100),
            nn.LeakyReLU(0.1)
        )

        # b*100->b*dim
        self.MLP_first = nn.Sequential(
            nn.Linear(100,self.max_dim)
        )

        # b*100->b*(2*dim)->b*dim
        self.MLP_second1 = nn.Sequential(
            nn.Linear(100,self.max_dim),
            nn.ReLU(),
        )
        self.MLP_second2 = nn.Linear(2*self.max_dim,self.max_dim)

        self.MLP_edge1 = nn.Sequential(
            nn.Linear(100,self.max_dim),
            nn.ReLU(),
        )
        self.MLP_edge2 = nn.Linear(3*self.max_dim,self.bond_types)

        self.MLP_stop = nn.Sequential(
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,2)
        )

        self.MLP_value = nn.Sequential(
            nn.Linear(100,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )


    def forward(self,N, A):

        H = self.GCN(N, A)
        '''特征融合'''
        H = torch.einsum('btji,btjk->btik', [H, H])


        H = self.conv1(H.unsqueeze(1))

        '''中间特征'''
        H = self.MLP(H.view(H.size(0),-1))

        value = self.MLP_value(H)

        out1 = self.MLP_first(H)

        # 加mask
        mask_first = Mask_first(N, self.max_atom_num)
        out1_masked = (mask_first.type(torch.float) - 1.) * 1000 + out1
        out_first = F.softmax(out1_masked, dim=1)
        first = Categorical(out_first).sample()


        out2 = self.MLP_second1(H)
        out2 = torch.cat([out1_masked,out2],dim=1)
        out2 = self.MLP_second2(out2)

        # 加mask
        mask_second = Mask_second(N, first.unsqueeze(1))
        out2_masked = (mask_second.type(torch.float) - 1.) * 1000 + out2
        out_second = F.softmax(out2_masked, dim=1)
        second = Categorical(out_second).sample()

        oute = self.MLP_edge1(H)
        oute = torch.cat([oute,out1_masked,out2_masked],dim=1)
        oute = self.MLP_edge2(oute)

        out_edge = F.softmax(oute, dim=1)  # batch*edge_type
        edge = Categorical(out_edge).sample()

        # 最后为stop
        out_stop = F.softmax(self.MLP_stop(H), dim=1)
        stop = Categorical(out_stop).sample()

        # 如果需要再过一层one hot encoding
        action = (first, second, edge, stop)
        output = (out_first, out_second, out_edge, out_stop)

        return output,action,value















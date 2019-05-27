from torch import nn
from gcn.basic_model import BasicModel
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

'''前向传播,H:初始原子的embeddings(Batch*n*k),E:对应不同化学键的邻接tensor(Batch*bond_type*n*n)'''


class Gconv1(nn.Module):
    def __init__(self, embed_dim, act=None, bond_type=4):
        super(Gconv1, self).__init__()
        self.act = act
        self.Weight = Parameter(torch.Tensor(bond_type, embed_dim, embed_dim))
        # parameter initialize
        init.xavier_normal_(self.Weight)

    '''图卷积层,输入预处理之后的图的L矩阵'''

    def forward(self, H, D, E):
        return self.act(torch.einsum('btij,btjk,btkl,btlm,tmd->btid', D, E, D, H, self.Weight))


'''计算公式 act(D(-1)AHW+HW)'''


class RGCNconv(nn.Module):
    def __init__(self, embed_dim, act=F.sigmoid, bond_type=4):
        super(RGCNconv, self).__init__()
        self.RW = Parameter(torch.Tensor(embed_dim, embed_dim, bond_type))
        self.SW = Parameter(torch.Tensor(embed_dim, embed_dim))
        init.xavier_normal_(self.RW)
        init.xavier_normal_(self.SW)
        self.act = act

    '''H:b*n*f  E:b*n*n*k'''

    def forward(self, H, E):
        D = torch.div(1, (E + torch.eye(E.size(1))).sum(2).sum(1))
        return self.act(
            torch.einsum('bi,bijk,bjm,mnk->bin', [D, E, H, self.RW]) + torch.einsum('bij,jl->bil', H, self.SW))


class Gconv_k_hops(nn.Module):
    def __init__(self, embed_dim, act=None, bond_type=4, k_hop=3):
        super(Gconv_k_hops, self).__init__()
        self.act = act


class RGCN(nn.Module):
    def __init__(self, config):
        super(RGCN, self).__init__()
        self.embedim = config.node_feature_dim
        self.bond_type = config.max_bond_type
        self.conv_act = nn.Sigmoid()
        self.agg = torch.mean

        '''network'''
        self.RGconv1 = RGCNconv(self.embedim, self.conv_act, self.bond_type)
        self.RGconv2 = RGCNconv(self.embedim, self.conv_act, self.bond_type)
        self.RGconv3 = RGCNconv(self.embedim, self.conv_act, self.bond_type)

        self.MLP = nn.Sequential(
            nn.Linear(self.embedim, 20),
            nn.LeakyReLU(0.1),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, H, E):
        H = self.RGconv1(H, E)
        H = self.RGconv2(H, E)
        H = self.RGconv3(H, E)
        score = self.agg(self.MLP(H))
        return score

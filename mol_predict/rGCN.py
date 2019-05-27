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
        D = torch.div(1, (E + torch.eye(E.size(2))).sum(2).sum(1))
        H = self.act(torch.einsum('bi,bkij,bjm,mnk->bin', [D, E, H, self.RW]) + torch.einsum('bij,jl->bil', H, self.SW))
        return H

class Gconv_k_hops(nn.Module):
    def __init__(self, embed_dim, act=None, bond_type=4, k_hop=3):
        super(Gconv_k_hops, self).__init__()
        self.act = act


class RGCN(BasicModel):
    def __init__(self, config):
        super(RGCN, self).__init__(config)
        self.embedim = config.node_feature_dim
        self.bond_type = config.max_bond_type
        self.conv_act = nn.Tanh()
        self.agg = torch.mean
        self.max_atom_num = config.max_atom_num

        '''network'''
        self.atom_embed = nn.Embedding(self.max_atom_num+1,self.embedim,padding_idx=0)
        self.RGconv1 = RGCNconv(self.embedim, self.conv_act, self.bond_type)
        self.RGconv2 = RGCNconv(self.embedim, self.conv_act, self.bond_type)
        self.RGconv3 = RGCNconv(self.embedim, self.conv_act, self.bond_type)
        # self.RGconv4 = RGCNconv(self.embedim, self.conv_act, self.bond_type)
        # self.RGconv5 = RGCNconv(self.embedim, self.conv_act, self.bond_type)
        # self.RGconv6 = RGCNconv(self.embedim, self.conv_act, self.bond_type)

        # self.MLP = nn.Sequential(
        #     nn.Linear(self.embedim, 1),
        #     nn.Sigmoid()
            # nn.Linear(30,1),

        # )
        self.MLP_act = nn.Tanh()
        self.MLP1 = nn.Linear(self.embedim,20)
        self.MLP2 = nn.Linear(20,20)
        self.MLP3 = nn.Linear(20,1)

    def forward(self, N, E):
        H = self.atom_embed(N)
        H = self.RGconv1(H, E)+H
        H = self.RGconv2(H, E)+H
        H = self.RGconv3(H, E)+H
        # H = self.RGconv4(H, E) + H
        # H = self.RGconv5(H, E) + H
        # H = self.RGconv6(H, E) + H
        # score = torch.mean(self.MLP(H),dim=1)
        H = self.MLP_act(self.MLP1(H))
        H = self.MLP_act(self.MLP2(H))+H
        score = torch.mean(self.MLP3(H),dim=1)

        return score

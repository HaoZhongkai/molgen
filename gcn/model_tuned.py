from torch import nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import init


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

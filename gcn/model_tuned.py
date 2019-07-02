from torch import nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F

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


class GCNconv(nn.Module):
    def __init__(self, embeddim, act=F.sigmoid, bond_type=3):
        super(GCNconv, self).__init__()
        self.act = act
        self.Weight = Parameter(torch.Tensor(bond_type, embeddim, embeddim))

        init.xavier_normal_(self.Weight)

    def forward(self, H, E):
        E = E + torch.eye(E.shape[2])
        D = torch.diag_embed(torch.sum(E, dim=3) ** (-1 / 2))  # 此时D为四维tensor(先通过reduce_sum降维再升维)

        return self.act(torch.einsum('btij,btjk,btkl,blm,tmd->btid', D, E, D, H, self.Weight)).sum(dim=1)


class RGCNkconv(nn.Module):
    def __init__(self, embeddim, k_hops, act=F.sigmoid, bond_type=3, use_gpu=True):
        super(RGCNkconv, self).__init__()
        self.act = act
        self.RWeight = []
        self.SWeight = []
        self.k_hops = k_hops
        self.use_gpu = use_gpu
        for i in range(k_hops):
            self.RWeight.append(Parameter(torch.Tensor(embeddim, embeddim, bond_type)))
            self.SWeight.append(Parameter(torch.Tensor(embeddim, embeddim)))
            init.xavier_normal_(self.RWeight[-1])
            init.xavier_normal_(self.SWeight[-1])

        self.k_MLP = nn.Linear(self.k_hops * embeddim, embeddim)
        if self.use_gpu:
            for i in range(k_hops):
                self.RWeight[i] = self.RWeight[i].cuda()
                self.SWeight[i] = self.SWeight[i].cuda()
            self.k_MLP.cuda()

    def forward(self, H, E):
        E = E + torch.eye(E.shape[2])
        E_k = E.clone()
        D = torch.div(1, E.sum(2).sum(1))
        # + torch.eye(E.size(2))
        H_ = self.act(torch.einsum('bi,bkij,bjm,mnk->bin', [D, E, H, self.RWeight[0]])
                      + torch.einsum('bij,jl->bil', H, self.SWeight[0]))
        for i in range(1, self.k_hops):
            E_k = torch.einsum('btij,btjk->btik', E_k, E)
            D_k = torch.div(1, (E_k + torch.eye(E.size(2))).sum(2).sum(1))
            H_k = self.act(torch.einsum('bi,bkij,bjm,mnk->bin', [D_k, E_k, H, self.RWeight[i]])
                           + torch.einsum('bij,jl->bil', H, self.SWeight[i]))
            H_ = torch.cat([H_, H_k], dim=2)

        H_ = self.act(self.k_MLP(H_))
        return H_

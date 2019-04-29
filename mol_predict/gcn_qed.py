from gcn.gcn0 import GCN_1
from torch import nn
from gcn.basic_model import BasicModel
import torch
import torch.nn.functional as F

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
            nn.BatchNorm2d(1),
            nn.Conv2d(1,3,4,2,1),   #feature/2
            nn.ReLU(),
            nn.Conv2d(3,5,4,2,1),   #feature/2
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.1)
        )
        self.linear1 = nn.Linear(int(5*(self.embeddim/4)**2),10)
        self.linear2 = nn.Linear(10,1)


    def forward(self,N, E):
        H = self.GCN(N, E)


        x = torch.einsum('bji,bjk->bik',[H,H])
        x = self.conv(x.unsqueeze(1))
        x = F.relu(self.linear1(x.view(x.size(0),-1)))
        x = self.linear2(x)
        # x = torch.mean(self.MLP(H),dim=1)

        return x


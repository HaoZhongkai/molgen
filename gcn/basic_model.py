import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import time

class BasicModel(nn.Module):
    def __init__(self,config):
        super(BasicModel,self).__init__()
        self.path = config.MODEL_PATH
        self.Model_name = str(type(self))

    def save(self,name=None):
        if name is None:
            name = '/'+time.strftime('%m%d_%H_%M.pkl')
            name = self.path + name
        torch.save(self.state_dict(),name)
        return name

    def forward(self, *input):
        pass

    def load(self,name):
        self.load_state_dict(self.path+name)


'''aggregate function input:输入张量,agg_fun:聚集函数,沿batch后的维度聚集'''
def AGG(input,axis=1):

    return torch.max(input,dim=axis)[0]


'''前向传播,H:初始原子的embeddings(Batch*n*k),E:对应不同化学键的邻接tensor(Batch*bond_type*n*n)'''
class Gconv(nn.Module):
    def __init__(self,embed_dim,act=None):
        super(Gconv,self).__init__()
        self.act = nn.LeakyReLU(0.1)
        self.Weight = Parameter(torch.Tensor(embed_dim,embed_dim))
        #parameter initialize
        init.xavier_normal_(self.Weight)

    '''图卷积层,输入预处理之后的图的L矩阵'''
    def forward(self,H,D,E):
        return AGG(self.act(torch.einsum('btij,btjk,btkl,blm,md->btid',D,E,D,H,self.Weight)))




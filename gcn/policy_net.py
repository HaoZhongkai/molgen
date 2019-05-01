from .basic_model import BasicModel
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from Funcs import *
from torch.nn import init
from torch.nn.parameter import Parameter

'''aggregate function input:输入张量,agg_fun:聚集函数,沿batch后的维度聚集'''
def AGG(input,axis=1):

    return torch.sum(input,dim=axis)



'''前向传播,H:初始原子的embeddings(Batch*n*k),E:对应不同化学键的邻接tensor(Batch*bond_type*n*n)'''
class Gconv(nn.Module):
    def __init__(self,bond_type,embed_dim,act=None):
        super(Gconv,self).__init__()
        self.act = nn.Tanh()
        self.Weight = Parameter(torch.Tensor(bond_type,embed_dim,embed_dim))
        self.BatchWeight = Parameter(torch.rand(bond_type))
        #parameter initialize
        init.xavier_normal_(self.Weight)


    '''图卷积层,输入预处理之后的图的L矩阵'''
    def forward(self,H,D,E):
        H = self.act(torch.einsum('btij,btjk,btkl,blm,tmd->btid',D,E,D,H,self.Weight))
        H = torch.einsum('t,btij->bij',self.BatchWeight,H)
        # return AGG(self.act(torch.einsum('btij,btjk,btkl,blm,tmd->btid',D,E,D,H,self.Weight)))
        return H


'''与GCN1一致,方便改进'''
class GCN_2(nn.Module):
    def __init__(self,config):
        super(GCN_2,self).__init__()
        self.bond_type = len(config.possible_bonds)
        self.agg_fun = config.agg_fun
        self.embeddim = config.node_feature_dim
        self.act = config.activation
        self.maxdim = config.max_atom_num+len(config.possible_atoms)

        self.atom_embed = nn.Embedding(len(config.possible_atoms)+1,self.embeddim,padding_idx=0)
        self.gcn_layer1 = Gconv(self.bond_type,self.embeddim,self.act)

        self.gcn_layer2 = Gconv(self.bond_type,self.embeddim,self.act)
        self.gcn_layer3 = Gconv(self.bond_type,self.embeddim,self.act)
        self.bn1 = nn.BatchNorm1d(self.maxdim)
        self.bn2 = nn.BatchNorm1d(self.maxdim)
        self.bn3 = nn.BatchNorm1d(self.maxdim)

    '''输入N为node array,E为邻接矩阵,计算H:node embedding (batch*n*k)'''
    def forward(self,N, E):
        # D:Tensor,通过对E(batch*type*n*n)中对第四维求和再升维得到的batch*type*n*n的张量
        H = self.atom_embed(N)
        E = E + torch.eye(E.size(2))
        D = torch.diag_embed(torch.sum(E, dim=3) ** (-1 / 2))  # 此时D为四维tensor(先通过reduce_sum降维再升维)
        # 通过三个图卷积层
        H = self.gcn_layer1(H,D,E)
        H = self.bn1(H)
        H = self.gcn_layer2(H,D,E)
        H = self.bn2(H)
        H = self.gcn_layer3(H,D,E)
        H = self.bn3(H)
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
        self.GCN = GCN_2(config)

        #nips上的Policy net构造
        #MLP1为H(0:n)->{0,1}^(n+c) ( we need a mask)
        self.MLP1 = nn.Sequential(
            nn.Linear(self.embeddim,20,bias=False),
            nn.ReLU(),
            nn.Linear(20,1,bias=False),

        )
        #MLP2为H(0:n+c)->{0,1}^(n+c)
        #实际使用时,将第一个选择的embedding选出来沿embedding维度成和整个node一样再过MLP
        self.MLP2 = nn.Sequential(
            nn.Linear(2*self.embeddim,30,bias=False),
            nn.ReLU(),
            nn.Linear(30,10,bias=False),
            nn.ReLU(),
            nn.Linear(10,1,bias=False),
        )

        #MLP3为H(0:n+c)->{0,1}^(bond_type)
        #选择第一个,第二个的节点feature连接起来过这个
        self.MLP3 = nn.Sequential(
            nn.Linear(2*self.embeddim,50,bias=False),
            nn.ReLU(),
            nn.Linear(50,10,bias=False),
            nn.ReLU(),
            nn.Linear(10,self.bond_types,bias=False),

        )

        #预测stop
        self.MLP4_1 = nn.Sequential(
            nn.Linear(self.embeddim,10,bias=False),
            nn.ReLU()
        )

        self.MLP4_2 = nn.Sequential(
            nn.Linear(10,2,bias=False),
        )

        self.MLP5 = nn.Sequential(
            nn.Linear(self.embeddim*self.max_dim,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )

    def evaluate_actions(self, N, A, actions):
        action_prob,action,value = self.forward(N, A)
        #print(actions)
        a_f, a_s, a_b, a_t = actions[:,0], actions[:,1], actions[:,2], actions[:,3]
        a_f_prob, a_s_prob, a_b_prob, a_t_prob = action_prob
        c_f, c_s, c_b, c_t = Categorical(a_f_prob), Categorical(a_s_prob), Categorical(a_b_prob), Categorical(a_t_prob)
        a_f_logp, a_s_logp, a_b_logp, a_t_logp = c_f.log_prob(a_f), c_s.log_prob(a_s), c_b.log_prob(a_b), c_t.log_prob(a_t)
        entropy = c_f.entropy() + c_s.entropy() + c_b.entropy() + c_t.entropy()

        return value, (a_f_logp, a_s_logp, a_b_logp, a_t_logp), entropy

    '''policy net的前向传播,传入N(现在分子的原子list),A(adj matrix list),输出action'''
    def forward(self,N,A):

        #计算node embeddings
        H = self.GCN(N,A)

        embed_first = H         #batch*n*k
        out1 = self.MLP1(embed_first).squeeze(dim=2)         #batch*(n+c)*k->batch*(n+c)*1->batch*(n+c)

        #加mask
        mask_first = Mask_first(N,self.max_atom_num)
        out1 = (mask_first.type(torch.float)-1.)*10000+out1
        out_first = F.softmax(out1,dim=1)
        first = Categorical(out_first).sample()         #batch long

        #先将对应的embed取出来,repeat后连接起来
        real_first_ebd = torch.cat([H[i,first[i]].unsqueeze(0) for i in torch.arange(first.size(0))],dim=0) #batch*k

        #沿embedding维度连接起来
        embed_second = torch.cat([H,real_first_ebd.unsqueeze(1).repeat(1,N.size(1),1)],dim=2)
        out2 = self.MLP2(embed_second).squeeze(dim=2)

        #加mask
        mask_second = Mask_second(N,first.unsqueeze(1))
        out2 = (mask_second.type(torch.float)-1.)*10000+out2
        out_second = F.softmax(out2,dim=1)
        second = Categorical(out_second).sample()


        #过第三道MLP
        # real_second_ebd = torch.index_select(H,0,first)  #batch*k
        real_second_ebd = torch.cat([H[i,second[i]].unsqueeze(0) for i in torch.arange(second.size(0))],dim=0)  #batch*k
        edge_embed = torch.cat([real_first_ebd,real_second_ebd],dim=1)
        out_edge = F.softmax(self.MLP3(edge_embed),dim=1)                #batch*edge_type
        edge = Categorical(out_edge).sample()


        #最后为stop
        out_stop = F.softmax(self.MLP4_2(torch.sum(self.MLP4_1(H),dim=1)),dim=1)
        # out_stop = F.softmax(self.MLP4(H.view(H.size(0),-1)),dim=1)
        stop = Categorical(out_stop).sample()

        #最后为value
        out_value = self.MLP5(H.view(H.size(0),-1))

        #如果需要再过一层one hot encoding
        action = (first,second,edge,stop)
        output = (out_first,out_second,out_edge,out_stop)

        return output,action,out_value


















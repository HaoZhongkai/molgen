from .basic_model import BasicModel,Gconv
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from Funcs import *
import numpy as np

'''网络结构:MolGAN'''

class GraphConvolution(nn.Module):

    # 9,[128,64],4
    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=nn.Tanh()):
        # input : 16x4x9
        # adj : 16x4x9x9
        D = torch.diag_embed(1/torch.sum(adj + torch.eye(adj.size(2)),dim=3))
        adj = torch.einsum('bijk,bikl->bijl',[D,adj])
        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)

        hidden = torch.einsum('bijk,bikl->bijl', [adj, hidden])
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', [adj, output])
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)
        output2 = torch.Tensor(output)
        # if output2.max()>1e3 or output2.min()<-1e3 or  np.any(np.isnan(output2.detach().numpy())):
        #     print('12')
        return output



class GraphAggregation(nn.Module):
    # 64 128 3
    def __init__(self, in_features, out_features, m_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+m_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+m_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation=nn.LeakyReLU(0.1)):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)
        output2 = torch.Tensor(output)
        # if output2.max()>1e3 or output2.min()<-1e3 or np.any(np.isnan(output2.detach().numpy())):
        #     print('11111111111111')
        return output


class Generator(nn.Module):
    '''Generator Default:
        Conv dims:[128,256,512]
        z dim: 8

     '''
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output) \
            .view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits





class PolicyNet(nn.Module):
    """Discriminator network with PatchGAN Default:
        d conv:[[128, 64], 128, [128, 64]]
        m dim: atom type nums 9
        b dim: bond num types 3
    ."""

    def __init__(self, config):
        super(PolicyNet, self).__init__()

        self.layer_config = config.rgcn_config
        self.max_dim = config.max_atom_num + len(config.possible_atoms)
        self.bond_types = len(config.possible_bonds)
        self.atom_types = len(config.possible_atoms)
        self.max_atom_num = config.max_atom_num


        graph_conv_dim, aux_dim, linear_dim = self.layer_config['conv_dim']
        m_dim = self.layer_config['m_dim']
        b_dim = self.layer_config['b_dim']
        dropout = self.layer_config['dropout']



        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, m_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim] + linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        # self.output_layer = nn.Linear(linear_dim[-1], 1)


        self.MLP_f = nn.Sequential(
            nn.Linear(linear_dim[-1],32,bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(32,self.max_dim,bias=False)
        )


        self.MLP_s = nn.Sequential(
            nn.Linear(linear_dim[-1],32,bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(32,self.max_dim,bias=False)
        )

        self.MLP_e = nn.Sequential(
            nn.Linear(linear_dim[-1],32,bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(32,self.bond_types,bias=False)
        )

        self.MLP_v = nn.Sequential(
            nn.Linear(linear_dim[-1],32),
            nn.LeakyReLU(0.1),
            nn.Linear(32,16),
            nn.LeakyReLU(0.1),
            nn.Linear(16,1)
        )


        self.MLP_t = nn.Sequential(
            nn.Linear(linear_dim[-1],32),
            nn.LeakyReLU(0.1),
            nn.Linear(32,2)
        )




    '''N为非onde hot 形式'''
    def forward(self,N, adj, hidden=None, activatation=None):

        #截取前9维
        node = N[:,:self.max_atom_num]
        adj = adj[:,:,:self.max_atom_num,:self.max_atom_num]

        node = one_hot(node,self.atom_types+1)

        annotations = torch.cat((hidden, node), -1) if hidden is not None else node

        h = self.gcn_layer(annotations, adj)

        annotations = torch.cat((h, hidden, node) if hidden is not None \
                                    else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        out1 = self.MLP_f(h)
        out2 = self.MLP_s(h)
        oute = self.MLP_e(h)
        outs = self.MLP_t(h)
        value = self.MLP_v(h)


        '''对输出加mask并采样'''
        # 加mask
        mask_first = Mask_first(N, self.max_atom_num)
        out1 = (mask_first.type(torch.float) - 1.) * 10000 + out1
        out_first = F.softmax(out1, dim=1)
        try:
            first = Categorical(out_first).sample()  # batch long
        except Exception:
            pass
        # 加mask
        mask_second = Mask_second(N, first.unsqueeze(1))
        out2 = (mask_second.type(torch.float) - 1.) * 10000 + out2
        out_second = F.softmax(out2, dim=1)
        second = Categorical(out_second).sample()



        out_edge = F.softmax(oute, dim=1)  # batch*edge_type
        edge = Categorical(out_edge).sample()



        # 最后为stop
        out_stop = F.softmax(outs, dim=1)
        # out_stop = F.softmax(self.MLP4(H.view(H.size(0),-1)),dim=1)
        stop = Categorical(out_stop).sample()


        # 如果需要再过一层one hot encoding
        action = (first, second, edge, stop )
        output = (out_first, out_second, out_edge, out_stop)



        return output,action,value



    def evaluate_actions(self, N, A, actions):
        action_prob,action,value = self.forward(N, A)
        #print(actions)
        a_f, a_s, a_b, a_t = actions[:,0], actions[:,1], actions[:,2], actions[:,3]
        a_f_prob, a_s_prob, a_b_prob, a_t_prob = action_prob
        c_f, c_s, c_b, c_t = Categorical(a_f_prob), Categorical(a_s_prob), Categorical(a_b_prob), Categorical(a_t_prob)
        a_f_logp, a_s_logp, a_b_logp, a_t_logp = c_f.log_prob(a_f), c_s.log_prob(a_s), c_b.log_prob(a_b), c_t.log_prob(a_t)
        entropy = c_f.entropy() + c_s.entropy() + c_b.entropy() + c_t.entropy()

        return value, (a_f_logp, a_s_logp, a_b_logp, a_t_logp), entropy
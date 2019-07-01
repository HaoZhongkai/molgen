import torch
from torch import nn
from gcn.basic_model import BasicModel
from config import Config
from gcn.model_tuned import RGCNconv
import torch.nn.functional as F

default_config = Config()

'''VAE encoder part, includes many graph convolution layers(currrently use R-GCN)'''
class Encoder(BasicModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.embeddim = config.node_feature_dim
        self.bond_type = config.max_bond_type
        self.agg = torch.mean
        self.max_atom_num = config.max_atom_num
        self.encoder_layers = config.encoder_layers
        self.res = config.res_connection
        self.load_path = config.vae_path['encoder']

        self.RGconv = []
        self.embedding_layer = nn.Embedding(self.max_atom_num + 1, self.embeddim, padding_idx=0)

        for i in range(self.encoder_layers):
            self.RGconv.append(RGCNconv(self.embeddim, bond_type=self.bond_type))

        # load if necessary
        if self.load_path:
            self.load_state_dict(torch.load(self.load_path))

    '''N :node feature, E:adj_matrix'''

    def forward(self, N, E):

        H = self.embedding_layer(N)

        for i in range(self.encoder_layers):
            H = H + self.RGconv[i](H, E) if self.res else self.RGconv[i](H, E)

        return H


'''VAE property predictor part,current params:[embeddim 40 20 1] with load options'''


class Predictor(BasicModel):
    def __init__(self, config, pred_id='qed'):
        super(Predictor, self).__init__(config)
        self.embeddim = config.node_feature_dim
        self.agg = torch.mean
        self.load_path = config.pred_path[pred_id]

        self.layer = nn.Sequential(
            nn.Linear(self.embeddim, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        # load if necessary
        if self.load_path:
            self.load_state_dict(torch.load(self.load_path))

    def forward(self, H):
        self.agg(self.layer(H))


'''VAE part decoder'''


class Decoder(BasicModel):
    def __init__(self, config):
        super(Decoder, self).__init__(config)
        self.embeddim = config.node_feature_dim
        self.max_atom_type = len(config.possible_atoms)
        self.max_atom_num = config.max_atom_num
        self.max_bond_type = config.max_bond_type
        self.load_path = config.vae_path['decoder']

        self.MLP_mu = nn.Linear(self.embeddim, self.embeddim)
        self.MLP_logvar = nn.Linear(self.embeddim, self.embeddim)

        self.Node_MLP = nn.Sequential(
            nn.Linear(self.embeddim, 25),
            nn.ReLU(),
            nn.Linear(25, self.max_atom_type + 1),
            nn.Softmax(dim=-1)
        )

        self.Link_MLP1 = nn.Linear(2 * self.embeddim, 40)
        self.Link_MLP2 = nn.Linear(40, 2)

        self.Edge_MLP1 = nn.Linear(2 * self.embeddim, 40)
        self.Edge_MLP2 = nn.Linear(40 + 40, self.max_bond_type)



        if self.load_path:
            self.load_state_dict(torch.load(self.load_path))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def rec_link(self, H):
        # at last use a sigmoid represents probs
        links = torch.Tensor([H.size(0), self.max_atom_num, self.max_atom_num, 2])
        edges = torch.Tensor([H.size(0), self.max_atom_num, self.max_atom_num, self.max_bond_type])

        # 先使用两个embedding 连接过MLP，再反过来concat求和除2，得到link的概率,
        # 过edge时先用对称过MLP，再连接link中间层embedding过MLP得到edge
        for i in range(self.max_atom_num - 1):
            for j in range(self.max_atom_num - 1):
                H_p0 = torch.cat([H[:, i], H[:, j]], dim=-1)
                H_n0 = torch.cat([H[:, j], H[:, i]], dim=-1)

                H_lp, H_ln = self.Link_MLP1(H_p0), self.Link_MLP1(H_n0)
                H_l = F.relu((H_ln + H_lp) / 2)
                link_logits = F.softmax(self.Link_MLP2(H_l), dim=-1)
                links[:, [i, j], [j, i]] = link_logits

                H_e = F.relu((self.Edge_MLP1(H_p0) + self.Edge_MLP1(H_n0)) / 2)
                H_e = torch.cat([H_e, H_l], dim=-1)
                edge_logits = F.softmax(self.Edge_MLP2(H_e), dim=-1)
                edges[:, [i, j], [j, i]] = edge_logits

        return links, edges

    def rec_loss(self, N, E, logits, weights=None):

        links = torch.sum(E.permute(0, 2, 3, 1), dim=-1).long()
        edges = torch.argmax(E.permute(0, 2, 3, 1), dim=-1, keepdim=False) * links

        # calculate node classification loss
        nodes_loss = F.cross_entropy(logits['node'], N, reduction='sum', reduce=True)

        # calculate link classification loss
        link_loss = F.cross_entropy(logits['link'].view(-1, 2), links, reduction='sum', reduce=True) / 2

        # calculate edge classification loss, only cal where edge exists
        edge_loss = torch.sum(F.cross_entropy(logits['edge'].view(-1, self.max_bond_type),
                                              edges.view(-1, self.max_bond_type), reduce=False).reshape_as(
            links) * links) / 2

        loss = nodes_loss + link_loss + edge_loss

        return loss

    def kl_loss(self, mu, logvar):

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return kl_loss

    def forward(self, H, N, E):
        mu = self.MLP_mu(H)
        logvar = self.MLP_logvar(H)

        H = self.reparameterize(mu, logvar)

        node_logits = self.Node_MLP(H)

        link_logits, edge_logits = self.rec_link(H)

        logits = {'node': node_logits, 'link': link_logits, 'edge': edge_logits}
        loss = self.kl_loss(mu, logvar) + self.rec_loss(N, E, logits)
        return logits,loss



'''the whole model of our VAE'''


class VAE(BasicModel):
    def __init__(self, config):
        super(VAE, self).__init__(config)
        self.embeddim = config.node_feature_dim
        self.bond_type = config.max_bond_type
        self.max_atom_num = config.max_atom_num
        self.res = config.res_connection
        self.pred_id = config.predictor_id
        self.pred_num = len(config.predictor_id)
        self.prop_weight = config.prop_loss_weight
        self.pro_criterion = nn.MSELoss()
        self.vis_loss = nn.L1Loss()

        '''calculating different kinds of loss function'''
        self.pred_on = True
        self.decoder_on = False

        self.encoder = Encoder(config)

        if self.pred_on:
            self.predictors = {pred_id: Predictor(config) for pred_id in self.pred_id}

        if self.decoder_on:
            self.decoder = Decoder(config)

    '''returns {}'''

    '''labels : {propid:tensor batch*1}'''
    def forward(self, N, E, labels):

        H = self.encoder(N, E)  # H after embeddings
        mol_graphs, props = [], []

        props_loss, de_loss, visloss = 0, 0, 0
        if self.pred_on:
            props = {predid: self.predictors[predid](H) for predid in self.pred_id}
            props_loss, visloss = self.pro_loss(props, labels)[0]

        if self.decoder_on:
            logits, de_loss = self.decoder(H, N, E)

        loss = props_loss + de_loss

        return {'props': props, 'graphs': mol_graphs, 'loss': loss, 'visloss': visloss}



    '''labels {'property_name':value(tensor[batch*1])} return loss for optimization and for visualization'''

    def pro_loss(self, props, labels):

        pro_loss = 0
        pro_visloss = {}
        for pro_id in labels.keys():
            pro_loss += self.prop_weight[pro_id] * self.pro_criterion(props[pro_id], labels[pro_id])
            pro_visloss.update({pro_id: self.vis_loss(props[pro_id], labels[pro_id])})

            '''left space for visualization'''

        return pro_loss, pro_visloss






    def save_components(self, id_list):
        if 'encoder' in id_list:
            self.encoder.save(name='encoder')
        if 'decoder' in id_list and self.decoder_on:
            self.decoder.save(name='decoder')
        if 'predictor' in id_list and self.pred_on:
            for key in self.pred_id:
                self.predictors[key].save(name=key)

        return

import torch
from torch import nn
from gcn.basic_model import BasicModel
from config import Config
from gcn.model_tuned import RGCNconv

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
        self.load_path = config.vae_path['decoder']

        self.MLP_mu = nn.Linear(self.embeddim, self.embeddim)
        self.MLP_sigma = nn.Linear(self.embeddim, self.embeddim)

        if self.load_path:
            self.load_state_dict(torch.load(self.load_path))

    def forward(self, H):
        pass


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

    def forward(self, N, E, labels):

        H = self.encoder(N, E)  # H after embeddings
        mol_graphs, props = [], []

        props_loss, re_loss, visloss = 0, 0, 0
        if self.pred_on:
            props = {predid: self.predictors[predid](H) for predid in self.pred_id}
            props_loss, visloss = self.pro_loss(props, labels)[0]

        if self.decoder_on:
            graphs = []

        loss = props_loss + re_loss

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

    def re_loss(self):
        pass

    def save_components(self, id_list):
        if 'encoder' in id_list:
            self.encoder.save(name='encoder')
        if 'decoder' in id_list and self.decoder_on:
            self.decoder.save(name='decoder')
        if 'predictor' in id_list and self.pred_on:
            for key in self.pred_id:
                self.predictors[key].save(name=key)

        return

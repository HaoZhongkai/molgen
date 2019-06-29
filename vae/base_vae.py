import torch
from torch import nn
from gcn.basic_model import BasicModel
from config import Config
from gcn.model_tuned import RGCNconv

default_config = Config()


class Encoder(BasicModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.embeddim = config.node_feature_dim
        self.bond_type = config.max_bond_type
        self.agg = torch.mean()
        self.max_atom_num = config.max_atom_num

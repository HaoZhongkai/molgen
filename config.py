import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from rdkit import Chem
import os


class Config():
    def __init__(self):

        # hyper parameter
        self.batch_size = 50
        self.lr = 1e-3
        self.lr_decay = 0.95
        self.max_epoch = 40




        # GCN options
        self.agg_fun = torch.mean
        self.activation = F.leaky_relu
        self.node_feature_dim = 40


        # Env options
        self.max_atom_num = 9
        self.use_scaffold = False
        self.use_random_sample = False
        self.possible_atoms = ['C','N','O','F']
        self.num_scaff = len(self.possible_atoms)
        self.dim = self.max_atom_num + self.num_scaff
        self.possible_bonds = [Chem.rdchem.BondType.SINGLE,
                               Chem.rdchem.BondType.DOUBLE,
                               Chem.rdchem.BondType.TRIPLE]
        self.max_bond_type = len(self.possible_bonds)
        self.add_scaffold_to_adj = True
        self.max_action_num = 30
        self.min_action_num = 4



        #file path
        self.PATH = os.path.dirname(__file__)
        self.DATASET_PATH = self.PATH + '/dataset/datasets'
        self.MODEL_PATH = self.PATH + '/dataset/models_'
        self.LOGS_PATH = self.PATH + '/dataset/logs'
        self.env_log_path = ''
        self.save_env_path = ''



        #pretrain sample
        self.num_qm9_trajectory = 5000
        self.pre_train_tra_save_path = '/home/jeffzhu/MCTs/dataset/datasets'


        #trajectory
        self.max_tr_per_molecule = 2
        self.random_tra_save_path = self.DATASET_PATH

        #molecule predict model options
        self.use_gpu = True
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD

        #visulize
        self.print_feq = 50



        #reward type
        self.reward_type = 'qed'
        self.final_valid_reward = 0
        self.final_not_valid_reward = -5
        self.reward_ratio = {'logP':0,'qed':2}
        self.step_reward = {'positive':1,'negative':0}
        self.early_stop = -0.1
        self.cut_out_tra = False


        #debug writer
        self.writer = SummaryWriter(log_dir=self.LOGS_PATH)



        #RGCN config
        self.rgcn_config = {
            'conv_dim':[[128, 64], 128, [128, 64]],
            'm_dim': len(self.possible_atoms)+1,
            'b_dim': len(self.possible_bonds),
            'dropout':False
        }

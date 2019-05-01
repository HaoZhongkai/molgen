from gcn import PolicyNet
from Funcs import *
from config import Config
import torch
from env.pg_env2 import Mole_Env

'''具体rollout使用的环境为mol_env而非mcts env(因为有reward设计),所有的rollout分子共用一个rollout环境'''
class RollOut():
    def __init__(self,config=Config(),policy_net=None):
        self.opt = config
        if policy_net:
            self.policy = policy_net
        else:
            self.policy = PolicyNet(config)
        self.env = Mole_Env(self.opt)


    #传入一个MC leaf(视具体情况可以换成mol),经过多次steps得出result
    def steps(self,leaf):

        mol = leaf.state
        terminal = False
        node_arr,adj,_,_ = self.env.inits(mol)
        reward = 0
        node_arr, adj = self.to_data(mol)

        #在非terminal的时候与env交互
        while not terminal:
            node_arr,adj = torch.Tensor(node_arr).type(torch.long),torch.Tensor(adj)


            #从policy调用forward以及一个解包函数解出action
            action = self.policy(node_arr,adj).get_action()
            node_arr,adj,reward_step,terminal = self.env.step(action)
            reward += reward_step

        self.postprocess()
        self.env.inits()

        return reward







    '''mol分子转换为data,tensor形式'''
    def to_data(self,mol):
        node_arr, adj = mol2data(mol,self.opt.max_atom_num,self.opt.possible_atoms,self.opt.possible_bonds,
                                 self.opt.add_scaffold_to_adj)
        node_arr = torch.Tensor(node_arr).type(torch.long).unsqueeze(0)
        adj = torch.Tensor(adj).unsqueeze(0)
        return node_arr,adj


    '''未来可能加入的后处理,如搜集当前分子状态加入buffer等'''
    def postprocess(self):
        pass
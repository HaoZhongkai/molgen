import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem import AllChem
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import copy
from rdkit.Chem.QED import qed
import time
import config
import sys
from Funcs import *

'''分子生成环境
    数据流法:原子与键更新在mol上
    mol->netx->node_arr,adj
    一次允许单个分子
    当动作不合理时,将保持原有分子不变
    将在环境中加入分子的reward
    
'''

class Mole_Env():
    def __init__(self,config):
        self.log_file_path = config.env_log_path
        self.save_path = config.save_env_path
        self.max_atom_num = config.max_atom_num  # 最大原子个数
        self.total_dim = config.dim  # 加上官能团后的总维度
        self.use_scaffold = config.use_scaffold  # 是否准备官能团
        self.use_random_sample = config.use_random_sample  # use random initialize
        self.batch = config.batch_size
        self.add_scaffolds = True


        # 可能的原子与化学键种类
        self.possible_atoms = config.possible_atoms
        self.max_atom_type = len(config.possible_atoms)  # 最大原子种类数
        self.possible_bonds = config.possible_bonds
        self.max_bond_type = len(config.possible_bonds)  # 价键的种类数
        self.max_action = config.max_action_num         #最大操作次数
        self.min_action = config.min_action_num


        self.reward_ratio = config.reward_ratio
        self.reward_type = config.reward_type
        self.reward_step_positive = config.step_reward['positive']
        self.reward_step_negative = config.step_reward['negative']
        self.early_terminal_reward = config.early_stop
        self.final_valid_reward = config.final_valid_reward
        self.final_not_valid_reward = config.final_not_valid_reward

        self.steps = 0
        self.inits()


    #原子初始化
    def inits(self,mol=None):

        if mol:
            self.mol = mol

        elif self.use_random_sample:
            self.mol = self.random_sample()
        else:
            self.mol = Chem.RWMol(Chem.MolFromSmiles('C'))  # 初始化一个C

        self.old_mol = copy.deepcopy(self.mol)
        self.graph = mol2nx(self.mol)
        self.node_arr, self.adj = mol2data(self.mol,self.max_atom_num,self.possible_atoms,self.possible_bonds,
                                           self.add_scaffolds)
        self.steps = 0
        return self.node_arr,self.adj,[],[]


    '''相互作用,action:(a_first,a_second,a_edge,a_terminal),输入时直接传进去具体数(非one hot)'''
    def step(self,action,force_final=False):
        a_first = action[0]
        a_second = action[1]
        a_edge = action[2]
        terminal = action[3]
        total_num = self.mol.GetNumAtoms()



        self.steps = self.steps+1
        self.old_mol = copy.deepcopy(self.mol)


        pass_test = True
        reward_step = 0
        reward_property = 0

        #当达到最大操作次数时或者本来terminal为true时,将terminal设为true
        if self.steps<=self.min_action:
            terminal = False
        elif self.steps<=self.min_action and terminal is True:
            reward_step += self.early_terminal_reward

        elif self.steps>self.max_action or terminal==1:
            terminal = True




        #terminal情况
        if not terminal:
            if a_second>=self.max_atom_num:
                if total_num>=self.max_atom_num:
                    terminal = True

                else:
                    self.add_atom(self.mol,a_second-self.max_atom_num)
                    a_second = total_num

                    #对于a_first或者a_second选择不当的,可能需要调用负反馈
                    if not self.add_bond(self.mol,a_first,a_second,a_edge):
                        # self.log('add bond wrong,negative reward')
                        pass_test = False

            elif a_second<total_num:
                if not self.add_bond(self.mol,a_first,a_second,a_edge):
                    # self.log('add bond wrong,negative reward')
                    pass_test = False

            else:
                # self.log('bond with no atoms,negative reward')
                pass_test = False
            #stop triggered



            if pass_test and self.val_check(self.mol) :

                '''通过中间步骤的化合价测试'''
                reward_step += self.reward_step_positive/self.max_atom_num


                self.update()
            else:
                # 未通过测试,不更新graph,返回


                reward_step +=self.reward_step_negative/self.max_atom_num
                # self.log('Step{0} valency test failed!'.format(self.steps))
                # self.log('Current atom Smile:' + Chem.MolToSmiles(self.old_mol))
                self.mol = copy.deepcopy(self.old_mol)


        elif terminal or force_final is True:

            # self.log('Stop Sign Triggered')
            '''计算最后的reward'''
            if self.chemical_check(self.mol):
                # 通过了valency check
                # print('Chemical Check Pass')
                reward_step += self.final_valid_reward
            else:

                reward_step -= self.final_not_valid_reward

            '''计算属性值奖励'''
            reward_property = self.reward_property(self.mol,self.reward_type,self.reward_ratio)


        reward_step += reward_property
        info = {'reward_step':reward_step,'qed':reward_property,'smiles':Chem.MolToSmiles(self.mol)}


        # for debug only
        #print(reward_step)

        return self.node_arr,self.adj,info,terminal












    def reward_property(self,mol,reward_type,reward_ratio):
        reward = 0
        if reward_type is 'qed':
            reward = qed(mol)*reward_ratio['qed']

        return reward







    '''(单个)分子添加原子'''
    def add_atom(self, mol, atom_id):
        atom_symbol = self.possible_atoms[atom_id]
        mol.AddAtom(Chem.Atom(atom_symbol))
        return


    '''(单个)分子添加化学键'''
    def add_bond(self, mol, a_first, a_second, a_edge):

        bond_type = self.possible_bonds[a_edge]
        bond = mol.GetBondBetweenAtoms(int(a_first), int(a_second))
        if bond or a_first == a_second:
            return False
        else:
            try:
                mol.AddBond(int(a_first), int(a_second), bond_type)
            except Exception:
                return False
            return True
        # except Exception:
        #     return

    '''根据mol更新其余参数(并返回)'''
    def update(self):
        self.graph = mol2nx(self.mol)
        self.node_arr, self.adj = mol2data(self.mol, self.max_atom_num, self.possible_atoms,self.possible_bonds,
                                           self.add_scaffolds)
        return self.node_arr,self.adj





    def chemical_check(self,mol):
        return chemical_validity_check(mol)

    def val_check(self,mol):
        return valency_check(mol)

    def random_sample(self):
        pass

    def log(self, msg='', data=True):
        print(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(msg) if data else str(msg))

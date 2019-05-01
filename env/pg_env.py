import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem import AllChem
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import copy
import time
import config
import sys
from Funcs import *

'''分子生成环境
    数据流法:原子与键更新在mol上
    mol->netx->node_arr,adj
    一次允许单个分子
    当动作不合理时,将保持原有分子不变
'''

class Pg_Env():
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


        self.steps = 0
        self.inits()


    #原子初始化
    def inits(self):


        if self.use_random_sample:
            self.mol = self.random_sample()
        else:
            self.mol = Chem.RWMol(Chem.MolFromSmiles('C'))  # 初始化一个C

        self.old_mol = copy.deepcopy(self.mol)
        self.graph = mol2nx(self.mol)
        self.node_arr, self.adj = mol2data(self.mol,self.max_atom_num,self.possible_atoms,self.possible_bonds,
                                           self.add_scaffolds)
        return self.node_arr,self.adj,[],[]


    '''相互作用,action:(a_first,a_second,a_edge,a_terminal)'''
    def step(self,action):
        a_first = np.argmax(action[0])
        a_second = np.argmax(action[1])
        a_edge = np.argmax(action[2])
        total_num = self.mol.GetNumAtoms()

        self.steps = self.steps+1
        self.old_mol = copy.deepcopy(self.mol)

        stop = False
        pass_test = True


        #terminal情况
        if action[3]==0:
            if a_second>=self.max_atom_num:
                self.add_atom(self.mol,a_second-self.max_atom_num)
                a_second = total_num



                #对于a_first或者a_second选择不当的,可能需要调用负反馈
                if self.add_bond(self.mol,a_first,a_second,a_edge):
                    self.log('add bond wrong,negative reward')
                    pass_test = False

                elif a_second<total_num:
                    if self.add_bond(self.mol,a_first,a_second,a_edge):
                        self.log('add bond wrong,negative reward')
                        pass_test = False

                else:
                    self.log('bond with no atoms,negative reward')
                    pass_test = False
            #stop triggered
            else:
                stop = True
                self.log('Stop Sign Triggered')

            if pass_test and self.val_check(self.mol) and self.chemical_check(self.mol):

                self.update()
            else:
                # 未通过测试,不更新graph,返回
                self.log('Step{0} Chemical validity test failed!'.format(self.steps))
                self.log('Current atom Smile:' + Chem.MolToSmiles(self.old_mol))
                self.mol = copy.deepcopy(self.old_mol)


            '''reward函数待更新'''
            reward = 0
            return self.node_arr,self.adj,reward












    def reward(self,mol):
        pass



    '''(单个)分子添加原子'''
    def add_atom(self, mol, atom_id):
        atom_symbol = self.possible_atoms[atom_id]
        mol.AddAtom(Chem.Atom(atom_symbol))
        return


    '''(单个)分子添加化学键'''
    def add_bond(self, mol, a_first, a_second, a_edge):
        bond_type = self.possible_bonds[a_edge]
        bond = mol.GetBondBetweenAtoms(int(a_first), int(a_second))
        if bond:
            return False
        else:
            try:
                mol.AddBond(int(a_first), int(a_second), bond_type)
            except Exception:
                return False
            return True


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

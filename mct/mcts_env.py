from Funcs import *
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from config import Config
from copy import deepcopy


'''为MCTS搜索树设计的molecule env,加快速度,整个MCTS搜索树只需共用一个MCTenv'''

class MCTS_Env1():
    def __init__(self,config):
        self.max_atom_num = config.max_atom_num  # 最大原子个数
        self.total_dim = config.dim  # 加上官能团后的总维度
        self.use_scaffold = config.use_scaffold  # 是否准备官能团

        # 可能的原子与化学键种类
        self.possible_atoms = config.possible_atoms
        self.max_atom_type = len(config.possible_atoms)  # 最大原子种类数
        self.possible_bonds = config.possible_bonds
        self.max_bond_type = len(config.possible_bonds)  # 价键的种类数
        self.terminal_bond_num = config.max_action_num


        self.check_action = True                #在确认为possible action之前检查化学有效性
        self.stop_atom_num_sign = self.max_atom_num-1     #分子大于等于这个原子个数以后在action中考虑stop



    '''获取所有可能的动作,输出为list(action),action为tuple(a_first,a_second,a_edge,a_terminal)
        当分子小于一定规模时,a_terminal默认为0,
        在确认为一个possible action之前需要经过chemical check
    '''
    def get_possible_actions(self,mol):
        possible_actions = []
        trial_mol = deepcopy(mol)
        atom_indexes = [atom.GetIdx() for atom in mol.GetAtoms()]
        stop_flag = False
        #当大于8个原子时可以考虑stop
        if len(atom_indexes)>=self.stop_atom_num_sign:
            stop_flag = True

        #下面分别考虑两种action模式,1.a_second为新加一个原子及一根键2.a_second为在内部加一个环
        for i in atom_indexes:
            for j in range(self.max_atom_type):
                for k in range(self.max_bond_type):
                    trial_mol = deepcopy(mol)
                    trial_mol.AddAtom(Chem.Atom(self.possible_atoms[j]))
                    trial_mol.AddBond(i,trial_mol.GetNumAtoms-1,self.possible_bonds[k])
                    if chemical_validity_check(trial_mol):
                        possible_actions.append((i,j+self.max_atom_num,k,0))
                        if stop_flag:
                            possible_actions.append((i,j+self.max_atom_num,k,1))


        #第二种action模式,未来可能会废弃
        for i in atom_indexes:
            for j in atom_indexes:
                for k in range(self.max_bond_type):
                    trial_mol = deepcopy(mol)
                    #如果加边成功进行后面操作
                    if self.add_bond(trial_mol,i,j,k):
                        if chemical_validity_check(trial_mol):
                            possible_actions.append((i,j,k,0))
                            if stop_flag:
                                possible_actions.append((i,j,k,1))

        return possible_actions


    '''对当前分子take action得到下一个分子'''
    def take_action(self,mol,action):

        if action[1]>=self.max_atom_num:
            try:
                mol.AddAtom(Chem.Atom(self.possible_atoms[action[1]-self.max_atom_num]))
                mol.AddBond(action[0],action[1]-self.max_atom_num,self.possible_bonds[action[2]])
            except Exception:
                print('Take Action Error:Invalid type 1:a_second>max_atoms')
                return mol
        elif action[2]<mol.GetNumAtoms():
            try:
                self.add_bond(mol,*action)

            except Exception:
                print('Take Action Error:Invalid type 2:a_second<mol_atoms')
                return mol
        else:
            print('Take Action Error:Invalid type 3:no atom on a_second')

        return mol




    #根据原子与键的数量判断是否终止
    def is_terminal(self,mol):
        if mol.GetNumAtoms()>self.max_atom_num or mol.GetNumBonds()>=self.terminal_bond_num:
            return True
        else:
            return False



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
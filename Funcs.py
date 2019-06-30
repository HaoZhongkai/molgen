import networkx as nx
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torchnet import meter

bond_type = [Chem.rdchem.BondType.SINGLE,
             Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE]



'''几个转换函数'''

'''由分子生成nx网络'''
def mol2nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def nx2mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        '''***************'''
        mol.AddBond(ifirst, isecond, bond_type)

    # Chem.SanitizeMol(mol)
    return mol



'''把networkx按化学键类型转化为邻接矩阵bond_type*n*n(n为最大原子数)'''
def nx2adj(G,max_atom_num,possible_bonds):
    #初始化邻接矩阵
    adj_mat = np.zeros([len(possible_bonds),max_atom_num,max_atom_num])
    bond_list = nx.get_edge_attributes(G,'bond_type')
    #查找对应的键,并修改邻接矩阵
    for edge in G.edges():
        first, second = edge
        bond_in_adj = possible_bonds.index(bond_list[first,second])
        adj_mat[[bond_in_adj],[first,second],[second,first]] = 1
    return adj_mat



'''把networkx按原子按one-hot形式转换为原子node array'''
def nx2node(G,max_atom_num,possible_atoms):

    node_array = np.zeros([max_atom_num,len(possible_atoms)])

    for node in G.nodes():
        node_array[node,possible_atoms.index(Chem.Atom(G.nodes[node]['atomic_num']).GetSymbol())] = 1
    return node_array


'''把networkx按原子非one hot 形式转换'''
def nx2node_(G,max_atom_num,possible_atoms):
    node_array = np.zeros([max_atom_num])
    for node in G.nodes():
        node_array[node] = possible_atoms.index(Chem.Atom(G.nodes[node]['atomic_num']).GetSymbol())+1
    return node_array


def visulize_mol(mols,savename=None):
    plt.ion()
    if type(mols) is list:

        fig = Draw.MolsToImage(mols)
    else:

        fig = Draw.MolToImage(mols)

    fig.show()
    if savename:
        fig.save(savename)
    print('okok')
    plt.ioff()
    return


'''根据action修改state'''
def renew_state(possible_atoms,max_atom_num,node_arr,adj_mat,action):
    # 非batch情况
    atom1 = np.argmax(action[0])
    atom2 = np.argmax(action[1])
    edge = np.argmax(action[2])
    current_num = len(np.nonzero(node_arr))
    if atom2>=max_atom_num:
        adj_mat[edge,[atom1,current_num],[current_num,atom1]] = 1
        node_arr[current_num] = atom2-max_atom_num

    elif atom2<current_num:
        adj_mat[edge,[atom1,atom2],[atom2,atom1]] = 1
    else:
        print('error add atom or edge on wrong place')

    return node_arr,adj_mat




#将adj转化为加入scaffold的adj
#允许接受单个或者一个batch
def adj_add_scaffold(adj,possible_atom_types):
    #考虑了batch情况
    if adj.ndim==4:
        new_adj = np.zeros([adj.shape[0],adj.shape[1],adj.shape[2]+possible_atom_types,adj.shape[2]+possible_atom_types])
        new_adj[:,:,:adj.shape[2],:adj.shape[2]] = adj

    else :
        new_adj = np.zeros([adj.shape[0],adj.shape[1]+possible_atom_types,adj.shape[1]+possible_atom_types])
        new_adj[:,:adj.shape[1],:adj.shape[1]] = adj

    return new_adj


#将单个或多个node array(非one hot形式)加上可能的原子
def node_add_scaffold(node_arr,possible_atom_types):
    #考虑batch情况:
    if node_arr.ndim==2:
        new_node_arr = np.zeros([node_arr.shape[0],node_arr.shape[1]+possible_atom_types])
        new_node_arr[:,:node_arr.shape[1]] = node_arr
        new_node_arr[:,node_arr.shape[1]:] = range(1,possible_atom_types+1)
    else:
        new_node_arr = np.zeros([node_arr.shape[0]+possible_atom_types])
        new_node_arr[:node_arr.shape[0]] = node_arr
        new_node_arr[node_arr.shape[0]:] = range(1,possible_atom_types+1)

    return new_node_arr





'''simles to adj and node_arr'''
def smiles2data(smiles,max_atom_num,atoms,bonds,batch=False):
    if batch or type(smiles) is list:
        error = 0
        adj = []
        node_arr = []
        for i in range(len(smiles)):
            try:
                net = mol2nx(Chem.MolFromSmiles(smiles[i]))
                adj.append(nx2adj(net,max_atom_num,bonds))
                node_arr.append(nx2node_(net,max_atom_num,atoms))
            except Exception:
                error +=1
                print('error',error)
        adj = np.stack(adj)
        node_arr = np.stack(node_arr)
        return node_arr,adj
    else:
        net = mol2nx(Chem.MolFromSmiles(smiles))
        adj = nx2adj(net,max_atom_num,bonds)
        node_arr = nx2node_(net,max_atom_num,atoms)
        return node_arr,adj



'''单个mol 转换为node_arr和adj_mat'''
def mol2data(mol,max_atom_num,possible_atoms,possible_bonds,add_sca=True):
    net = mol2nx(mol)
    adj = nx2adj(net,max_atom_num,possible_bonds)
    node_arr = nx2node_(net,max_atom_num,possible_atoms)
    if add_sca:
        adj = adj_add_scaffold(adj,len(possible_atoms))
        node_arr = node_add_scaffold(node_arr,len(possible_atoms))
    return node_arr,adj



'''检查(单个)分子化学有效性'''
def chemical_validity_check(mol):


    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False


'''分子的化合价检查0为不通过,1为通过'''
def valency_check(mol):
    try:
        Chem.SanitizeMol(mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False



'''对第一个不可选择的原子加mask,规则为只能从现有分子的原子中选择
    输入为结点矩阵Tensor:(batch*(n+c)) long,以及最大原子数n,对0到n中可能的最大原子数加mask
    mask: batch*(n+c) 0 or 1 long
'''
def Mask_first(N,max_atom_num):
    mask = torch.zeros(N.size()).type(torch.long)
    mask[:,:max_atom_num] = 1
    mask = torch.where(N==0,N,mask)
    return mask




'''对第二个不可选择的原子加mask,规则为只能从可选原子的分子中选择且不能与第一个相同
    N 结点矩阵Tensor,(batch*(n+c)) long,first choice:选择的第一个原子Tensor:(batch*1) long
    mask: batch*(n+c) 0 or 1 long
'''
def Mask_second(N,first_choice):
    mask = torch.ones(N.size(),dtype=torch.long)
    mask = torch.where(N==0,N,mask)
    mask = mask.scatter(1,first_choice.type(torch.long),0)
    return mask


#把 ...*n形式的long转化为...*n*dim的one hot形式
def one_hot(x,embedding_dim):
    return torch.zeros(*x.size(),embedding_dim).scatter(x.ndimension(),x.unsqueeze(-1),1)


'''提供一个处理多项loss的简单结构的meter'''


class MAvgMeter():
    def __init__(self, id_list):
        self.id_list = id_list
        self.meter = {ids: meter.AverageValueMeter() for ids in id_list}

    def reset(self):
        for key in self.id_list:
            self.meter[key].reset()

    def add(self, values):
        for key in self.id_list:
            self.meter[key].add(values[key])

    '''return mean value'''

    def value(self, key=None):
        if key:
            return self.meter[key].value
        else:
            return {key: self.meter[key].value for key in self.id_list}

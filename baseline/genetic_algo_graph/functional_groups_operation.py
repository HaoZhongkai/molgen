import networkx as nx
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import Draw
from Funcs import *
from random import *
from queue import Queue

vocab_nodes_decode = {
    1: 'C',
    2: 'N',
    3: ['O', 'S'],
    4: ['F', 'Cl', 'Br']
}

vocab_nodes_encode = {
    'C': 1, 'N': 2, 'O': 3, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4
}

vocab_bonds = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE
]
params = {
    'population_size': 2000,
    'max_atom_num': 30,
    'nodes_decode': vocab_nodes_decode,
    'nodes_encode': vocab_nodes_encode,
    'bonds': vocab_bonds,
    'property_kind': 'qed',
    'select_rate': 0.4,
    'mutate_rate': 0.5,
    'node_mutate_rate': 0.4,
    'select_annealing_rate': 0.8,
    'full_valence': 1 + 4,
    'fitnessfun': 'qed',
    'max_iter': 2000,
    'max_add_bond_attempt': 8,
    'max_add_atom_attempt': 8
}

table_of_elements = {
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    16: 'S',
    17: 'Cl',
    35: 'Br',
}

possible_bonds = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE
]


def adj2mol(nodes, adj, possible_bonds):
    mol = Chem.RWMol()

    for i in range(len(nodes)):
        atom = Chem.Atom(nodes[i])
        mol.AddAtom(atom)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if adj[i, j]:
                mol.AddBond(i, j, possible_bonds[adj[i, j] - 1])

    return mol


class SubGraphOperation(object):
    def __init__(self, params):
        self.population_size = params['population_size']  # ~1e3e4 population inits and the minima nums after crossover
        self.graph_size = params['max_atom_num']  # ~50
        self.vocab_nodes_decode = params['nodes_decode']
        self.vocab_nodes_encode = params['nodes_encode']
        self.possible_bonds = params['bonds']
        self.property_name = params['property_kind']
        self.mutate_rate = params['mutate_rate']
        self.node_mutate_rate = params['node_mutate_rate']
        self.selection_annealing_rate = params['select_annealing_rate']  # weed out good gene at annealing prob
        self.selection_rate = params['select_rate']  # ~1/2 of population,population may not be accurately the size
        self.full_valence = params['full_valence']  # ~5
        self.fitnessfun = params['fitnessfun']

    ''' given smiles representation, get the node list '''

    def get_node_list(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        G = mol2nx(mol)
        atomic_nums = nx.get_node_attributes(G, 'atomic_num')
        node_list = []
        for i in range(len(atomic_nums)):
            node_list.append(table_of_elements[atomic_nums[i]])
        return node_list

    ''' given smiles representation, get the adjacent matrix'''

    def get_adj_mat(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        G = mol2nx(mol)
        atomic_nums = nx.get_node_attributes(G, 'atomic_num')
        adj = np.zeros([len(atomic_nums), len(atomic_nums)])
        bond_list = nx.get_edge_attributes(G, 'bond_type')
        for edge in G.edges():
            first, second = edge
            adj[[first], [second]] = possible_bonds.index(bond_list[first, second]) + 1
            adj[[second], [first]] = possible_bonds.index(bond_list[first, second]) + 1
        return adj

    ''' get the diagonal matrix '''

    def get_diag_mat(self, node_list):
        length = len(node_list)
        diag_mat = np.zeros([length, length])
        for i in range(length):
            diag_mat[[i], [i]] = vocab_nodes_encode[node_list[i]]
        return diag_mat

    def get_expand_mat(self, adj, diag_mat):
        return adj + diag_mat

    ''' combine two subgraphs using graph representation , add subg2 to subg1 '''

    def combine_two_subgraph_G(self, subg1, subg2):
        len1, len2 = subg1.shape[0], subg2.shape[0]
        goal_adj = np.zeros([len1 + len2, len1 + len2])
        goal_adj[:len1, :len1] = subg1
        goal_adj[len1:, len1:] = subg2
        ''' this part can be modified '''
        row1, row2 = self.mask(subg1), self.mask(subg2)
        row2 = [i + len1 for i in row2]
        index1, index2 = choice(row1), choice(row2)
        goal_adj[index1, index2] = 1
        goal_adj[index2, index1] = 1
        return goal_adj

    ''' combine two subgraphs using smiles representation, add subg2 to subg1 '''

    def combine_two_subgraph_S(self, smiles1, smiles2):
        adj_mat1, adj_mat2 = self.get_adj_mat(smiles1), self.get_adj_mat(smiles2)
        node_list1, node_list2 = self.get_node_list(smiles1), self.get_node_list(smiles2)
        diag_mat1, diag_mat2 = self.get_diag_mat(node_list1, node_list2)
        exp_mat1, exp_mat2 = self.get_expand_mat(adj_mat1, diag_mat1), self.get_expand_mat(adj_mat2, diag_mat2)
        G = self.combine_two_subgraph_G(exp_mat1, exp_mat2)
        return G

    '''calculate the row which can continue to operate'''

    def mask(self, adj):
        node_num = np.count_nonzero(adj.diagonal())
        row_sum = np.sum(adj[:node_num, :node_num], axis=0)
        mask_row = np.argwhere(row_sum < self.full_valence).squeeze(axis=1).tolist()
        return mask_row

    ''' input the mol, output the picture '''

    def visual_fucntion(self, mol, filename='temp.png'):
        Draw.MolToFile(mol, filename)


# sgp = SubGraphOperation(params)
# node_list = sgp.get_node_list('C1CCCCC1')
# adj = sgp.get_adj_mat('C1CCCCC1')
# diag_mat = sgp.get_diag_mat(node_list)
# expand_mat = adj + diag_mat
# mask_row = sgp.mask(expand_mat)
#
# a = sgp.combine_two_subgraph_G(expand_mat,expand_mat)
# a = a.astype(int)
# print(a)
# n = node_list + node_list
# mol = adj2mol(n, a, vocab_bonds)
# sgp.visual_fucntion(mol)

class Foundation(object):
    def __init__(self, params, smiles):
        self.population_size = params[
            'population_size']  # ~1e3e4 population inits and the minima nums after crossover
        self.graph_size = params['max_atom_num']  # ~50
        self.vocab_nodes_decode = params['nodes_decode']
        self.vocab_nodes_encode = params['nodes_encode']
        self.possible_bonds = params['bonds']
        self.property_name = params['property_kind']
        self.mutate_rate = params['mutate_rate']
        self.node_mutate_rate = params['node_mutate_rate']
        self.selection_annealing_rate = params['select_annealing_rate']  # weed out good gene at annealing prob
        self.selection_rate = params['select_rate']  # ~1/2 of population,population may not be accurately the size
        self.full_valence = params['full_valence']  # ~5
        self.fitnessfun = params['fitnessfun']
        self.max_add_bond_attempt = params['max_add_bond_attempt']
        self.max_add_atom_attempt = params['max_add_atom_attempt']

        self.subgraphoperation = SubGraphOperation(params)
        self.smiles = smiles

        self.node_list = []  # ['C','N', ...]
        self.expand_mat = []
        self.max_bond_type = []
        self.index_in_moleclue = []  # the index of each atom in term of the whole molecule
        self.bond_type = []  # the bond type for each atom

        self.adj = self.subgraphoperation.get_adj_mat(self.smiles)
        self.adj = self.adj.astype(int)
        self.node_list = self.subgraphoperation.get_node_list(self.smiles)
        diag_mat = self.subgraphoperation.get_diag_mat(self.node_list)
        self.expand_mat = self.subgraphoperation.get_expand_mat(self.adj, diag_mat)
        self.expand_mat = self.expand_mat.astype(int)

        self.num_atom = len(self.node_list)
        self.index_in_moleclue = [i for i in range(self.num_atom)]
        self.bond_type = [0 for i in range(self.num_atom)]

    def updata_adj_mat(self):
        temp_expand_mat = self.expand_mat
        length = self.num_atom
        for i in range(length):
            temp_expand_mat[i, i] = 0
        self.adj = temp_expand_mat.astype(int)

    def _add_atom_between_bond(self):
        if self.num_atom < 2:
            return -1
        temp_elements = {
            0: 'C',
            1: 'N',
            2: 'O',
            3: 'S'
        }
        temp_num = np.random.choice([i for i in range(4)])
        atom = temp_elements[temp_num]
        # self.node_list.append(atom)

        flag = 0

        for i in range(self.max_add_atom_attempt):
            index = np.random.choice(len(self.node_list))
            row_ = list(np.squeeze(np.argwhere(self.adj[index] > 0), axis=1))
            if len(row_) > 0:
                flag = 1
                index_ = np.random.choice(row_, 1)
                self.num_atom += 1
                self.node_list.append(atom)
                temp_expand_mat = np.zeros([self.num_atom, self.num_atom])
                temp_expand_mat[:self.num_atom - 1, :self.num_atom - 1] = self.expand_mat
                temp_expand_mat[self.num_atom - 1, self.num_atom - 1] = vocab_nodes_encode[atom]
                temp_expand_mat[index, index_] = temp_expand_mat[index_, index] = 0
                temp_expand_mat[index, self.num_atom - 1] = temp_expand_mat[self.num_atom - 1, index] = 1
                temp_expand_mat[index_, self.num_atom - 1] = temp_expand_mat[self.num_atom - 1, index_] = 1
                self.expand_mat = temp_expand_mat
                self.updata_adj_mat()
                break
        return flag

    def _add_bond(self):
        flag = 0
        mask_row = self.subgraphoperation.mask(self.expand_mat)
        rest_row = mask_row
        rest_row_len = len(rest_row)

        if rest_row_len >= 2:
            for i in range(self.max_add_bond_attempt):
                index = np.random.choice(rest_row_len, 2, replace=False)
                node1, node2 = index[0], index[1]
                if self.expand_mat[node1, node2] >= 1:
                    self.expand_mat[node1, node2] += 1
                    self.expand_mat[node2, node1] += 1
                    flag = 1
                    self.updata_adj_mat()
                    break
        return flag


class FunctionalGroup(Foundation):
    def __init__(self, params, smiles):
        Foundation.__init__(self, params, smiles)


class Molecule(Foundation):
    def __init__(self, params, smiles):
        Foundation.__init__(self, params, smiles)

    '''There are three kinds of operations for mutation, add_atom, add_atom_between_bond, add_bond'''
    '''The two other operations are implemented in parent class Foundation'''

    def _add_atom(self):
        if self.num_atom < 2:
            return -1
        temp_elements = {
            0: 'C',
            1: 'N',
            2: 'O',
            3: 'S'
        }
        temp_num = np.random.choice([i for i in range(4)])
        atom = temp_elements[temp_num]

        flag = 0

        mask_row = self.subgraphoperation.mask(self.expand_mat)
        if len(mask_row) >= 1:
            flag = 1
            self.num_atom += 1
            self.node_list.append(atom)
            index = np.random.choice(mask_row, 1)
            temp_expand_mat = np.zeros([self.num_atom, self.num_atom])
            temp_expand_mat[:self.num_atom - 1, :self.num_atom - 1] = self.expand_mat
            temp_expand_mat[self.num_atom - 1, self.num_atom - 1] = vocab_nodes_encode[atom]
            temp_expand_mat[self.num_atom - 1, index] = temp_expand_mat[index, self.num_atom - 1] = 1
            self.expand_mat = temp_expand_mat
            self.updata_adj_mat()
        return flag


class Population(object):
    def __init__(self, params):
        self.population_size = params[
            'population_size']  # ~1e3e4 population inits and the minima nums after crossover
        self.graph_size = params['max_atom_num']  # ~50
        self.vocab_nodes_decode = params['nodes_decode']
        self.vocab_nodes_encode = params['nodes_encode']
        self.possible_bonds = params['bonds']
        self.property_name = params['property_kind']
        self.mutate_rate = params['mutate_rate']
        self.node_mutate_rate = params['node_mutate_rate']
        self.selection_annealing_rate = params['select_annealing_rate']  # weed out good gene at annealing prob
        self.selection_rate = params['select_rate']  # ~1/2 of population,population may not be accurately the size
        self.full_valence = params['full_valence']  # ~5
        self.fitnessfun = params['fitnessfun']
        self.max_add_bond_attempt = params['max_add_bond_attempt']
        self.max_add_atom_attempt = params['max_add_atom_attempt']

        self.mol1 = Molecule(params, 'CC(=O)NCCC1=CNC2=C1C=C(OC)C=C2')

    def init_population(self):

        pass

    '''following function is designed for '''

    def bfs_molecule(self, mol_test):
        length = mol_test.num_atom
        sigma = 0.15 * length
        mu = 0.67 * length
        '''using normal distribution to select the number of molecules'''
        num_sample = (int)(np.random.normal(loc=mu, scale=sigma))

        adj = mol_test.adj
        np.putmask(adj, adj >= 1, 1)
        row_sum = np.sum(adj, axis=0)
        index_start = np.argmax(row_sum)

        hash = np.zeros(length)
        res = []
        q = Queue(self.graph_size)
        q.put(index_start)
        res.append(index_start)
        hash[index_start] = 1

        while len(res) < num_sample:
            node = q.get()
            node_list = list(np.squeeze(np.argwhere(adj[node] >= 1), axis=1))
            for n in node_list:
                if hash[n] != 1:
                    hash[n] = 1
                    q.put(n)
                    res.append(n)

        temp_node_list = []
        '''res store the index of atom in the molecules'''
        for i in res:
            temp_node_list.append(mol_test.node_list[i])

        num_atom = len(res)
        temp_mat = np.zeros([num_atom, num_atom])
        print(res)
        for i in range(num_atom - 1):
            for j in range(1, num_atom):
                temp_mat[i, j] = mol_test.expand_mat[res[i], res[j]]
                temp_mat[j, i] = mol_test.expand_mat[res[j], res[i]]

        temp_mat = temp_mat.astype(int)
        length = len(temp_node_list)

        for i in range(length):
            temp_mat[i, i] = vocab_nodes_encode[temp_node_list[i]]
        # mol = adj2mol(temp_node_list, temp_mat, possible_bonds)
        # Draw.MolToFile(mol, 'test2.png')
        '''temp_mat for expand_mat, temp_node_list for node_list'''
        return temp_mat, temp_node_list

    '''the following function implements the crossover for molecules'''

    def crossover(self, mol1, mol2):
        temp_mat1, temp_node_list1 = self.bfs_molecule(mol1)
        temp_mat2, temp_node_list2 = self.bfs_molecule(mol2)
        sgo = SubGraphOperation(params)
        goal_mat = sgo.combine_two_subgraph_G(temp_mat1, temp_mat2).astype(int)
        goal_list = temp_node_list1 + temp_node_list2
        for i in range(len(goal_list)):
            goal_mat[i, i] = 0
        mol = adj2mol(goal_list, goal_mat, possible_bonds)
        Draw.MolToFile(mol, 'temp.png')

    def mutate(self, mol1):
        choice = np.random.choice(3, 1)
        if choice[0] == 0:
            mol1._add_atom()
        elif choice[0] == 1:
            mol1._add_atom_between_bond()
        else:
            mol1._add_bond()


# test_mol = Molecule(params, 'C1CCCC1')
# mol1 = adj2mol(test_mol.node_list, test_mol.adj, possible_bonds)
# Draw.MolToFile(mol1, 'test.png')
#
# test_mol._add_atom_between_bond()
# print(test_mol.node_list)
# mol2 = adj2mol(test_mol.node_list, test_mol.adj, possible_bonds)
# Draw.MolToFile(mol2, 'test2.png')

# functional_group = Molecule(params, "C1CCCC1")
# functional_group._add_atom_between_bond()
# functional_group._add_bond()
# functional_group._add_atom()
# #print(functional_group.expand_mat)
# length = len(functional_group.node_list)
# exp_mat = functional_group.expand_mat
# for i in range(length):
#     exp_mat[i, i] = 0
#
# exp_mat = exp_mat.astype(int)
#
# node_list = functional_group.node_list
# #print(functional_group.node_list)
# mol1 = adj2mol(node_list, exp_mat, possible_bonds)
# smile = Chem.MolToSmiles(mol1)
# print(smile)


mol1 = Molecule(params, 'CC(=O)NCCC1=CNc2c1cc(OC)cc2')
mol2 = Molecule(params, 'CC(=O)NCCC1=CNc2c1cc(OC)cc2')

pop1 = Population(params)
pop1.crossover(mol1, mol2)
mol_temp = Molecule(params, 'C1CCCC1')
pop1.mutate(mol_temp)
test = adj2mol(mol_temp.node_list, mol_temp.adj, possible_bonds)
Draw.MolToFile(test, 'test.png')

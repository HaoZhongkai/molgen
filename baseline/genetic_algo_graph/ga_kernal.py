import rdkit
import rdkit.Chem as Chem
import numpy as np
from Funcs import *
import copy
import random
import networkx as nx

'''genetic algorithm baseline working on graph
    add mask to each mutation to ensure the validity of molecule
    represent molecule in 'extended' adjcent matrix with following rule
    C-->1  N-->2  O-->3  F-->4  S-->2'
    mutation based on matrix vector space decomposition
'''
vocab = {
    '1': 'C',
    '2': ['N', 'Al'],
    '3': ['O', 'S'],
    '4': ['F', 'Cl', 'Br']
}

'''
    GA pipeline: inits-->mutation-->selection-->expandsion--|
                        <-----------------------------------|
'''

'''
    Population contains the basic operations include mutation,selection,inits and evaluation
    Individuals contains a extended molecule smiles, atom num, fitness, node feature extended adj val_flag
        data flow: extended adj-> nodes -> smiles -> adj (mutate on adj) 
'''


class Population():
    def __init__(self, params, inits_smiles=None):
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

        self.iter = 0
        self.population = []
        self.best_indi = []
        self.max_popu_size = self.population_size
        self.inits(inits_smiles)

    '''population inits contains at least an C atom, otherwise cause problem for the algorithm,
        first just init a C atom
    '''

    def inits(self, inits_smiles=None):
        for i in range(self.population_size):
            init_adj = np.zeros([self.graph_size, self.graph_size], dtype=int)
            init_adj[0, 0] = 1
            self.population.append(self.decode(copy.deepcopy(init_adj)))

        if inits_smiles:
            for smi in inits_smiles:
                for i in random.sample(range(self.population_size),
                                       int(1 / (2 * len(inits_smiles)) * self.population_size)):
                    self.population[i] = self.encode(smiles=smi)

        self.population_size = len(self.population)
        return

    '''now just init a C atom'''

    def init_mole(self):
        init_adj = np.zeros([self.graph_size, self.graph_size], dtype=int)
        init_adj[0, 0] = 1
        return self.decode(init_adj)

    ''' mutate on a graph
        rules: first sample several molecules execute the add-atom operations choosing from the basis
               second sample several molecules execute the internal superposition operation
        
    '''

    '''mutate the whole population, first mutate, then add node'''

    def mutate_popu(self):
        mutate_list1 = random.sample(range(self.population_size), int(self.mutate_rate * self.population_size))
        for i in mutate_list1:
            self.population[i] = self.mutation_inter(self.population[i])
        mutate_list2 = random.sample(range(self.population_size), int(self.mutate_rate * self.population_size))
        for i in mutate_list2:
            self.population[i] = self.add_node(self.population[i])

        return

    '''can use pre mask or post mask,now we use premask to change an edge and then with prob to change a node
        later may change mutation times per molecule
    '''

    def mutation_inter(self, indi):
        ava_rows = self.mask(indi)
        if len(ava_rows) > 1:
            mu_nodes = random.sample(ava_rows, k=2)
            if indi['adj'][mu_nodes[0], mu_nodes[1]] < 3:
                indi['adj'][[mu_nodes[0], mu_nodes[1]], [mu_nodes[1], mu_nodes[0]]] += 1
        ava_rows = self.mask(indi)
        if ava_rows and random.uniform(0, 1) < self.node_mutate_rate:  # to avoid too many other atoms
            mu_node = random.choice(ava_rows)
            if indi['adj'][mu_node, mu_node] < 4:
                indi['adj'][mu_node, mu_node] += 1

        # decode and evaluate
        indi = self.decode(indi['adj'])
        return indi

    '''try to randomly add node to add to graph and connect an edge'''

    def add_node(self, indi):
        nodes_num = indi['nodes_num']
        atom_index = 1
        atom_sign = self.get_node(atom_index)
        ava_rows = self.mask(indi)
        if len(ava_rows) > 0:  # have availble rows
            mutate_node = random.choice(ava_rows)
            indi['adj'][nodes_num, nodes_num] = atom_index
            indi['adj'][[nodes_num, mutate_node], [mutate_node, nodes_num]] = 1
            # to save computation resource, directly operate the mol
            indi['nodes'].append(atom_sign)
            indi['nodes_num'] += 1
            indi['mol'].AddAtom(Chem.Atom(atom_sign))
            indi['mol'].AddBond(nodes_num, int(mutate_node), Chem.BondType.SINGLE)  # seems need int not int64
            indi['val_flag'] = chemical_validity_check(indi['mol']) and valency_check(indi['mol'])
            if indi['val_flag']:
                indi['fitness'] = self.fitnessfun(indi['mol'])
                indi['smiles'] = Chem.MolToSmiles(indi['mol'])
            else:
                indi['fitness'] = 0
                indi['smiles'] = ''
        return indi

    '''selection
    '''

    def selection(self):
        p = self.selection_rate
        q = self.selection_annealing_rate

        fitnesses = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitnesses[i] = self.population[i]['fitness']
        fitness_bound = np.sort(fitnesses)[int((1 - p) * self.population_size)]
        rand_prob = np.random.uniform(0, 1, self.population_size)
        del_list = []
        for i in range(self.population_size):
            if (self.population[i]['fitness'] > fitness_bound and rand_prob[i] > p) or \
                    (self.population[i]['fitness'] < fitness_bound and rand_prob[i] > 1 - p):
                del_list.append(i)
        self.population = list(np.delete(self.population, del_list))
        self.population_size = len(self.population)
        self.iter += 1

        return

    '''could 1.sample from graph 2. copy now we use copy until the sampling and 
        combining algorithm is finished'''

    def popu_expand(self):
        expand_list = random.choices(range(self.population_size), k=self.max_popu_size - self.population_size)
        for i in expand_list:
            '''to avoid population crupt sample some and replace with C, with prob'''
            if random.uniform(0, 1) > 0.1:
                self.population.append(copy.deepcopy(self.population[i]))
            else:
                self.population.append(self.init_mole())
        self.population_size = len(self.population)

        return

    '''add mask seems just return a list of not saturated atom is ok'''

    def mask(self, indi):
        node_num = np.count_nonzero(indi['adj'].diagonal())
        row_sum = np.sum(indi['adj'][:node_num, :node_num], axis=0)
        mask_row = list(np.argwhere(row_sum < self.full_valence).squeeze(axis=1).tolist())
        return mask_row

    '''renew after a operation is done on a adj while other properties does not renew,return a indi'''

    def decode(self, adj, possible_bonds=None):
        # decode node feature
        node_num0 = np.count_nonzero(adj.diagonal())
        nodes = adj.diagonal()[:node_num0]
        nodes = [self.get_node(i) for i in nodes]
        bonds = possible_bonds if possible_bonds else self.possible_bonds
        mol = adj2mol(nodes, adj, bonds)
        if chemical_validity_check(mol):
            val_flag = True
            fitness = self.fitnessfun(mol)

        else:
            val_flag = False
            fitness = 0
        indi = {'adj': adj, 'nodes': nodes, 'nodes_num': node_num0, 'mol': mol, 'smiles': Chem.MolToSmiles(mol),
                'val_flag': val_flag, 'fitness': fitness}

        return indi

    def get_population_stats(self):
        fitnesses = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitnesses[i] = self.population[i]['fitness']
        best_fitness = fitnesses.max()
        best_index = fitnesses.argmax()
        best_indi = copy.deepcopy(self.population[best_index])
        self.best_individual = best_indi  # renew it in self
        avg_fitness = fitnesses.mean()
        effective_num = np.count_nonzero(fitnesses)

        return {
            'best_fitness': best_fitness,
            'best_individual': best_indi,
            'best_index': best_index,
            'avg_fitness': avg_fitness,
            'effective_num': effective_num,
            'fitnesses': fitnesses
        }

    '''might add weight to nodes'''

    def get_node(self, index):
        return random.choice(self.vocab_nodes_decode[int(index)])

    '''first just do a simple encode process, while later can be extended on molecule partly protected'''

    def encode(self, smiles=None, mol=None):
        G = mol2nx(mol) if mol else mol2nx(Chem.MolFromSmiles(smiles))
        adj = np.zeros([self.graph_size, self.graph_size], dtype=int)
        bond_list = nx.get_edge_attributes(G, 'bond_type')

        for edge in G.edges():
            first, second = edge
            adj[[first, second], [second, first]] = self.possible_bonds.index(bond_list[first, second]) + 1

        node_num = mol.GetNumAtoms()
        nodes = np.zeros(node_num)
        for node in G.nodes():
            nodes[node] = self.vocab_nodes_encode[Chem.Atom(G.nodes[node]['atomic_num']).GetSymbol()]
        if chemical_validity_check(mol):
            val_flag = True
            fitness = self.fitnessfun(mol)

        else:
            val_flag = False
            fitness = 0
        indi = {'adj': adj, 'nodes': nodes, 'nodes_num': node_num, 'mol': mol, 'smiles': Chem.MolToSmiles(mol),
                'val_flag': val_flag, 'fitness': fitness}

        return indi


'''nodes : input list of atoms like ['C', 'N']...,adj :extended or normal adj matrix, will not check'''


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

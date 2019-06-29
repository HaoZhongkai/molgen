import rdkit
import numpy as np
import copy
from rdkit import Chem as Chem
from rdkit.Chem.QED import qed as qed
from baseline.genetic_algo_graph.ga_kernal import Population
from tensorboardX import SummaryWriter

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
    'fitnessfun': qed,
    'max_iter': 2000
}

inits_smiles = ['CC(O)c1ccccc1', 'C1=CC=CN=C1', 'C[C@H](O)C1=CC=CC=C1', 'c1ccccc1', 'C1=CC=CC=C1OC']

population = Population(params)
writer = SummaryWriter(logdir='./logs')
stats = []
best_score = 0
best_smiles = []

for iter in range(params['max_iter']):
    population.mutate_popu()
    population.selection()
    population.popu_expand()

    stats = population.get_population_stats()
    best_score = max(best_score, stats['best_fitness'])
    best_smiles.append(stats['best_individual']['smiles'])
    writer.add_scalar('best_score', best_score)
    print(best_score, best_smiles[-1], stats['effective_num'])

print(best_score)
print('finish')

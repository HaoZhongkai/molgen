import rdkit
from baseline.genetic_algo_smiles.ga_kernal import Population as Population
from tensorboardX import SummaryWriter

# vocab = ('C','F','H','I','N','O','P','S','c','l','n','o','r','s','#','(',')','+', '-','/','1','2','3','4','5','6','7','8'
#     ,'=','@','B','[','\\',']',)
vocab = ('C', 'F', 'H', 'N', 'O', '#', '(', ')', '+', '-', '1', '2', '3', '=')

params = {
    'population_size': 500,
    'max_symbol_num': 30,
    'bit_symbol_num': 5,
    'vocab':vocab,
    'property_kind':'qed',
    'crossover_segment_rate': 0.1,
    'mutate_rate': 0.5,
    'mutate_gene_rate': 0.3,
    'select_annealing_rate': 1,
    'select_rate':0.5,
    'max_iter_num': 3000,
}
inits_smiles = ['CC(O)c1ccccc1', 'C1=CC=CN=C1', 'C[C@H](O)C1=CC=CC=C1', 'c1ccccc1', 'C1=CC=CC=C1OC']

population = Population(params)
writer = SummaryWriter(logdir='./logs')
stats = []
best_score = 0
best_smiles = []


for iter in range(params['max_iter_num']):
    population.crossover()
    population.mutate()
    population.selection()
    stats = population.get_population_stats()

    best_score = max(best_score, stats['best_fitness'])
    best_smiles.append(stats['best_individual']['smiles'])
    writer.add_scalar('best_score',best_score)
    print(best_score, best_smiles[-1], stats['effective_num'])

print(best_score)
print('finish')




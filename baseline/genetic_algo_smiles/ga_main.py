import rdkit
from baseline.genetic_algo_smiles.ga_kernal import Population as Population
from tensorboardX import SummaryWriter

vocab = ('C','F','H','I','N','O','P','S','c','l','n','o','r','s','#','(',')','+', '-','/','1','2','3','4','5','6','7','8'
    ,'=','@','B','[','\\',']',)


params = {
    'population_size':2000,
    'max_symbol_num':50,
    'bit_symbol_num':6,
    'vocab':vocab,
    'property_kind':'qed',
    'crossover_segment_rate':0.3,
    'mutate_rate':0.25,
    'mutate_gene_rate':0.2,
    'select_annealing_rate':0.8,
    'select_rate':0.5,
    'max_iter_num':300,
}

population = Population(params)
writer = SummaryWriter(logdir='./logs')
stats = []
best_score = []
best_smiles = []


for iter in range(params['max_iter_num']):
    population.crossover()
    population.mutate()
    population.selection()
    stats = population.get_population_stats()


    best_score.append(stats['best_fitness'])
    best_smiles.append(stats['best_individual'])
    writer.add_scalar('best_score',best_score)
    print(best_smiles)

print(best_score[-1])
print('finish')




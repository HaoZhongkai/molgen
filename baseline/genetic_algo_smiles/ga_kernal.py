import rdkit
import numpy as np
'''
    since the total vocabulary has 34 elements, then in the gene every 6 bits would be used to represent a character in
    the vocabulary
'''
vocab = ('C','F','H','I','N','O','P','S','c','l','n','o','r','s','#','(',')','+', '-','/','1','2','3','4','5','6','7','8'
    ,'=','@','B','[','\\',']',)



'''Population for genetic algorithm
    population_gene:list of gene as 1-d numpy array value 0 or 1
    population_fitness: list of the properties of the molecule of the population
     
'''
class Population():
    def __init__(self, params):
        self.population_size = params['population_size']  # ~1e3e4
        self.gen_num = params['max_symbol_num']  # ~50
        self.gene_len = params['bit_symbol_num']      #~6
        self.vocab = params['vocab']
        self.population = []
        self.population_smiles = []
        self.population_fitness = []

        self.inits()


    # init random 0 1 array as initial population
    def inits(self,init_population=None):
        if init_population:
            self.population = init_population
        else:

            indication_array = np.random.randint(1, self.gen_num, size=self.population_size)
            for i in range(self.population_size):
                self.population.append(np.zeros(self.gene_len * self.gen_num))
                self.population[-1][:indication_array[i]] = np.random.randint(2, size=indication_array[i])


        return



    def crossover(self):
        pass

    def mutation(self):
        pass

    def inversion(self):
        pass

    def selection(self):
        pass

    def eval(self):
        pass


    # convert gene to smiles representations decode the whole population
    def decode_genes(self):
        pass


    #decode individual genes
    def decode_gene(self, gene, gene_len, gene_num):
        index_set = []

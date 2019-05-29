import rdkit
from rdkit.Chem.QED import qed
from rdkit import Chem as Chem
import numpy as np
import copy
'''
    since the total vocabulary has 34 elements, then in the gene every 6 bits would be used to represent a character in
    the vocabulary
'''
vocab = ('C','F','H','I','N','O','P','S','c','l','n','o','r','s','#','(',')','+', '-','/','1','2','3','4','5','6','7','8'
    ,'=','@','B','[','\\',']',)



'''Population for genetic algorithm
    population_gene:list of gene as 1-d numpy array value 0 or 1
    population_fitness: list of the properties of the molecule of the population
    population: list of dicts contain population gene smiles and fitness
     
'''
class Population():
    def __init__(self,params):
        self.population_size = params['population_size']    #~1e3e4 population inits and the minima nums after crossover
        self.gene_num = params['max_symbol_num']      #~50
        self.gene_len = params['bit_symbol_num']      #~6
        self.vocab = params['vocab']
        self.property_name = params['property_kind']
        # self.crossover_rate = params['crossover_rate']
        self.crossover_segment_rate = params['crossover_segment_len']
        self.mutate_rate = params['mutate_rate']
        self.mutate_gene_rate = params['mutate_gene_rate']
        self.selection_annealing_rate = params['annealing_rate']
        self.selection_rate = params['selection_num']      #~1/2 of population,population may not be accurately the size

        self.chrome_len = self.gene_len*self.gene_num
        self.vocab_num = len(self.vocab)
        self.population = []
        self.inits()


    # init random 0 1 array as initial population
    def inits(self,init_population=None):
        if init_population:
            for indi in init_population:
                self.population.append({'gene':self.encode(indi),'smiles':indi,'fitness':0})
        else:

            indication_array = np.random.randint(1,self.gene_num,size=self.population_size)
            for i in range(self.population_size):
                self.population.append({'gene':np.zeros(self.gene_len*self.gene_num),'smiles':'','fitness':0})
                self.population[-1]['gene'][:indication_array[i]] = np.random.randint(2,size=indication_array[i])
        self.decode_genes()
        self.eval_fitness_all()
        return


    '''perform uniform crossover on genes,and add them to the population with its smiles and fitness updated
    , then populationsize  is updated'''
    def crossover(self):
        # num_pairs = int(self.crossover_rate*self.population_size)
        while len(self.population)<self.population_size:
            gene1 = copy.deepcopy(self.population[np.random.randint(self.population_size)]['gene'])
            gene2 = copy.deepcopy(self.population[np.random.randint(self.population_size)]['gene'])
            crossover_gene = np.random.choice(self.chrome_len,int(self.crossover_segment_rate*self.chrome_len))
            gene1[crossover_gene],gene2[crossover_gene] = gene2[crossover_gene],gene1[crossover_gene]
            smiles1,smiles2 = self.decode_gene(gene1), self.decode_gene(gene2)
            self.population.append({'gene':gene1,'smiles':smiles1,'fitness':self.eval_fitness(smiles1)})
            self.population.append({'gene':gene2,'smiles':smiles2,'fitness':self.eval_fitness(smiles2)})

        self.population_size = len(self.population)
        return


    '''mutate and update the same way not add new'''
    def mutate(self):
        num_mutates = int(self.mutate_rate*self.population_size)
        mutate_list = np.random.choice(self.population_size,num_mutates)
        for index in mutate_list:
            mutate_segment = np.random.choice(self.chrome_len,int(self.chrome_len*self.mutate_gene_rate))
            self.population[index]['gene'][mutate_segment] = 1-self.population[index]['gene'][mutate_segment]
            self.population[index]['smiles'] = self.decode_gene(self.population[index]['gene'])
            self.population[index]['fitness'] = self.eval_fitness(self.population[index]['smiles'])

        return


    def inversion(self):
        pass



    def selection(self):
        p = self.selection_annealing_rate
        q = self.selection_rate
        fitness_arr = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness_arr[i] = self.population[i]['fitness']
        fitness_bound = fitness_arr.sort()[int((1-q)*self.population_size)]
        # higher than the bound would be selection at prob p, else 1-p
        del_list = []
        rand_prob = np.random.uniform(0,1,self.population_size)

        for i in range(self.population_size):
            if (self.population[i]['fitness']>fitness_bound and rand_prob>p) or (self.population[i]['fitness']<
            fitness_bound and rand_prob>1-p):
                del_list.append(i)
        np.delete(self.population,del_list).tolist()
        self.population_size = len(self.population)
        return






    # evaluate fitness for all population
    def eval_fitness_all(self):
        if self.property_name is 'qed':
            eval_fun = qed
        else:
            eval_fun = None

        for i in range(len(self.population_size)):
            mol = self.chemical_check(self.population[i]['smiles'])
            self.population[i]['fitness'] = eval_fun(mol) if mol else 0

        return

    # for individual
    def eval_fitness(self,smiles):
        if self.property_name is 'qed':
            eval_fun = qed
        else:
            eval_fun = None
        mol = self.chemical_check(self.population[i]['smiles'])
        fitness = eval_fun(mol) if mol else 0
        return fitness


    # convert gene to smiles representations decode the whole population
    def decode_genes(self):
        for i in range(len(self.population)):
            self.population[i]['smiles'] = self.decode_gene(self.population[i]['gene'])

        return




    def encode(self,smiles):
        gene = np.zeros(self.gene_num*self.gene_len)
        for i in range(len(smiles)):
            bstr = bin(self.vocab.index(smiles[i]))[2:]
            gene[(i+1)*self.gene_len-len(bstr):(i+1)*self.gene_len] = np.fromstring(bstr)

        return gene



    #decode individual genes
    def decode_gene(self,gene):
        smiles = ''
        for i in self.gene_num:
            b = gene[i*self.gene_len:(i+1)*self.gene_len]
            index = b.dot(1<<np.arange(b.size)[::-1])
            smiles+=self.vocab[index] if index<self.vocab_num else ''
        return smiles


    # to save invoke times ,return mol when it is a valid molecule
    def chemical_check(self,smiles):
        if Chem.MolFromSmiles(smiles):
            return
        else:
            return False



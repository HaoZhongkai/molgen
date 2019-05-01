from data_pre.sparse_molecular_dataset import SparseMolecularDataset
from config import config
import rdkit.Chem as Chem
import numpy as np
from Funcs import *
import random
dconfig = config()

data = SparseMolecularDataset()
data.load('/home/jeffzhu/MCTs/dataset/datasets/gdb9_9nodes.sparsedataset')
smiles = random.sample(list(data.smiles),10000)
data = smiles2data(smiles,dconfig.max_atom_num,dconfig.possible_atoms,dconfig.possible_bonds)

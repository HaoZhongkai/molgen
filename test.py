import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.QED import qed


import networkx as nx
import argparse
import multiprocessing
from rdkit import Chem

# filename = 'dataset/datasets/zinc.smi'
# mol_set = [Chem.MolFromSmiles(line) for line in open(filename, 'r').readlines()]
# mol_set = mol_set[:500]
# w = Chem.SDWriter('test.sdf')
# for m in mol_set:
#     w.write(m)

data = Chem.SDMolSupplier('test.sdf')
for i in data:
    print(Chem.MolToSmiles(i))

# print(Chem.MolToSmiles(data[316],kekuleSmiles=True))
# # len_count = 0
# # for i in data:
# #     print(len_count)
# #     Chem.Kekulize(i)
# #     print(Chem.MolToSmiles(i, kekuleSmiles=True))
# #     len_count += 1
#     # w.write(Chem.MolFromSmiles(Chem.MolToSmiles(i, kekuleSmiles=True)))
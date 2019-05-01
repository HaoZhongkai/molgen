from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from Funcs import *
from config import config
dconfig = config()
def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

mol = Chem.MolFromSmiles('c1c[nH+]c2cc[nH]c2n1')
data = smiles2data('c1c[nH+]c2cc[nH]c2n1',dconfig.max_atom_num,dconfig.possible_atoms,dconfig.possible_bonds,batch=False)
Draw.MolToImage(mol_with_atom_index(mol)).show()
print(data)
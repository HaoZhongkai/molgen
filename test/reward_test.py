from Funcs import *
from env.pg_env2 import Mole_Env
from rdkit import Chem
from config import Config

dconfig = Config()

env_ = Mole_Env(dconfig)

action = [[0,1+9,1,0],
          [0,2+9,0,0],
          [0,0,1,0],
          [2,3,0,0],
          [0,0,0,1]]
out = []
for i in range(5):
    out.append(env_.step(action[i]))

print('ok')
print('ok')







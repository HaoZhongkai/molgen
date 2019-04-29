import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import Config
from env.pg_env2 import *
from ppo import *

config = Config()
env = Mole_Env(config)
ppo = PPO(env)

batch_size = 50
ppo_iteration = 8

#for name,_ in ppo.net.named_parameters():

while True:
    obs, actions, values, neglogpacs, Gt, sum = ppo.interaction()
    print(sum)
    if len(ppo.actions) > 100:
        index = np.random.choice(len(ppo.actions), batch_size, replace=False)
        obs_, actions_, values_, neglogpacs_, Gt_ = [], [], [], [], []
        for i in index:
            obs_.append(ppo.obs[i])
            actions_.append(ppo.actions[i])
            values_.append(ppo.values[i])
            neglogpacs_.append(ppo.neglogpacs[i])
            Gt_.append(ppo.Gt[i])
        for i in range(ppo_iteration):
            ppo.update(obs_, Gt_, actions_, values_, neglogpacs_, 0.2)
            print(ppo.net.GCN.gcn_layer3.Weight.grad)
        ppo.reset()
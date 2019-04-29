import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import Config
from env.pg_env2 import *
from ppo.ppo import PPO

config = Config()
env = Mole_Env(config)
ppo = PPO(env)

ppo.interaction()
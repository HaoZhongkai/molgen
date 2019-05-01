import torch
import torch.nn as nn
from torch.distributions import Categorical
from ppo import *
import gym

env = gym.make('CartPole-v0')

ppo = PPO_Discrete(env)

while True:
    obs, actions, values, neglogpacs, Gt, sum = ppo.interaction()
    obs_ = torch.Tensor(obs)
    actions_ = torch.Tensor(actions)
    values_ = torch.Tensor(values)
    neglogpacs_ = torch.Tensor(neglogpacs)
    Gt_ = torch.Tensor(Gt)

    print(sum)

    for i in range(10):
        ppo.update(obs_, Gt_, actions_, values_, neglogpacs_, 0.2)
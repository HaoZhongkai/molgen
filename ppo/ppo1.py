import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

import numpy as np
import time
import torch.optim as optim
import torch
from torchnet import meter
from gcn.policy_net import *
from config import Config
from tensorboardX import SummaryWriter
import copy

class PPO(object):
    def __init__(self, env,config):
        self.env = env
        self.net = PolicyNet(Config())
        self.gamma = 0.98

        self.optimizer = optim.SGD(self.net.parameters(), 0.01)

        self.obs = []
        self.actions = []
        self.values = []
        self.neglogpacs = []
        self.rewards = []
        self.Gt = []
        self.writer = config.writer
        self.iter = 0

        self.max_buffer_size = 1000

    def sample(self, batch_size):
        if len(self.obs) > self.max_buffer_size:
            self.clean_buffer()
        index = np.random.choice(len(self.obs), batch_size)

        temp_obs, temp_actions, temp_values, temp_neglogpacs, temp_Gt = [],[],[],[],[]
        for i in index:
            temp_obs.append(self.obs[i])
            temp_actions.append(self.actions[i])
            temp_values.append(self.values[i])
            temp_neglogpacs.append(self.neglogpacs[i])
            temp_Gt.append(self.Gt[i])
        return temp_obs,temp_actions,temp_values,temp_neglogpacs,temp_Gt


    def reset(self):
        self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.Gt = [],[],[],[],[],[]

    #action在这里是四元组，action_Prob也是四元组
    def interaction(self):
        temp_obs, temp_actions, temp_values, temp_neglogpacs, temp_reward = [],[],[],[],[]
        sum = 0
        N, A, _, _ = self.env.inits()

        info = {'reward':0,'qed':0,'smiles':''}
        while(True):
            N_ = torch.Tensor(N[np.newaxis,:]).long()
            A_ = torch.Tensor(A[np.newaxis,:])
            with torch.no_grad():
                action_prob, action, value = self.net(N_, A_)

            a_f_p, a_s_p, a_b_p, a_t_p = action_prob
            c_f, c_s, c_b, c_t = Categorical(a_f_p), Categorical(a_s_p), Categorical(a_b_p), Categorical(a_t_p)
            a_f, a_s, a_b, a_t = action
            a_f_logp, a_s_logp, a_b_logp, a_t_logp = c_f.log_prob(a_f), c_s.log_prob(a_s), c_b.log_prob(a_b), c_t.log_prob(a_t)

            action_np = (a_f.numpy()[0], a_s.numpy()[0], a_b.numpy()[0], a_t.numpy()[0])
            action_prob_np = (a_f_logp.numpy()[0], a_s_logp.numpy()[0], a_b_logp.numpy()[0], a_t_logp.numpy()[0])
            obs_np = (N, A)

            N_next, A_next, rew, done = self.env.step(action_np)

            info['reward'] += rew['reward_step']
            # sum += rew['']

            temp_obs.append(obs_np)
            temp_actions.append(action_np)
            # self.values.append(value.numpy()[0][0])
            temp_values.append(float(value.numpy()[0]))
            temp_neglogpacs.append(action_prob_np)
            temp_reward.append(rew['reward_step'])

            if done:
                R = 0
                index = rew['best_index']
                info['qed'] = rew['qed']
                info['smiles'] = rew['smiles']

                temp_obs = temp_obs[:index]
                temp_actions = temp_actions[:index]
                temp_values = temp_values[:index]
                temp_neglogpacs = temp_neglogpacs[:index]
                temp_reward = temp_reward[:index]

                self.obs += temp_obs
                self.actions += temp_actions
                self.values += temp_values
                self.neglogpacs += temp_neglogpacs
                self.rewards += temp_reward

                for i in reversed(self.rewards):
                    R = R * self.gamma + i
                    self.Gt.insert(0, R)
                break
            else:
                N, A = N_next, A_next

        return temp_obs, temp_actions, temp_values, temp_neglogpacs, self.Gt, info

    def clean_buffer(self):
        length = len(self.obs)

        temp_obs = copy.deepcopy(self.obs)
        temp_actions = copy.deepcopy(self.actions)
        # self.values.append(value.numpy()[0][0])
        temp_values = copy.deepcopy(self.values)
        temp_neglogpacs = copy.deepcopy(self.neglogpacs)
        temp_reward = copy.deepcopy(self.rewards)
        temp_Gt = copy.deepcopy(self.Gt)

        self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.Gt = [],[],[],[],[],[]

        index_arr = np.random.choice(length, int(length/2))

        for i in index_arr:
            self.obs.append(temp_obs[i])
            self.actions.append(temp_actions[i])
            self.values.append(temp_values[i])
            self.neglogpacs.append(temp_neglogpacs[i])
            self.rewards.append(temp_reward[i])
            self.Gt.append(temp_Gt[i])

    def update(self, obs, returns, actions, values, neglogpacs, cliprange_now):
        returns = torch.Tensor(returns).view(-1,1)
        values = torch.Tensor(values).view(-1,1)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = Variable(advantages).view(-1,1)

        N, A = [], []
        for obs in obs:
            N.append(obs[0])
            A.append(obs[1])
        N = Variable(torch.Tensor(N).long())
        A = Variable(torch.Tensor(A))

        actions = Variable(torch.Tensor(actions))
        values = Variable(values).view(-1, 1)
        returns = Variable(returns).view(-1, 1)
        advantages = Variable(advantages).view(-1, 1)

        vpred, action_log_probs, dist_entropy = self.net.evaluate_actions(N, A, actions)

        old_a_f_prob, old_a_s_prob, old_a_b_prob, old_a_t_prob = [],[],[],[]
        for item in neglogpacs:
            old_a_f_prob.append(item[0])
            old_a_s_prob.append(item[1])
            old_a_b_prob.append(item[2])
            old_a_t_prob.append(item[3])
        old_a_f_prob, old_a_s_prob, old_a_b_prob, old_a_t_prob = torch.Tensor(old_a_f_prob), torch.Tensor(old_a_s_prob), torch.Tensor(old_a_b_prob), torch.Tensor(old_a_t_prob)
        old_a_f_prob, old_a_s_prob, old_a_b_prob, old_a_t_prob = Variable(old_a_f_prob.view(-1,1)), Variable(old_a_s_prob.view(-1,1)), Variable(old_a_b_prob.view(-1,1)), Variable(old_a_t_prob.view(-1,1))
        a_f_prob, a_s_prob, a_b_prob, a_t_prob = action_log_probs
        a_f_prob, a_s_prob, a_b_prob, a_t_prob = Variable(a_f_prob.view(-1,1)), Variable(a_s_prob.view(-1,1)), Variable(a_b_prob.view(-1,1)), Variable(a_t_prob.view(-1,1))

        # ratio = torch.exp(a_f_prob+a_s_prob+a_b_prob+a_t_prob - old_a_f_prob - old_a_s_prob - old_a_b_prob - old_a_t_prob)
        ratio = torch.exp(a_f_prob+a_s_prob+a_b_prob - old_a_f_prob - old_a_s_prob - old_a_b_prob )

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange_now, 1.0 + cliprange_now) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0
        # value_loss = (returns - vpred).pow(2).mean()

        self.optimizer.zero_grad()
        (value_loss + action_loss - dist_entropy.mean() * 0.01).backward()
        # nn.utils.clip_grad_norm(self.net.parameters(), 0.5)
        self.optimizer.step()

        '''visulize'''
        self.iter += 1
        # if self.iter %10 ==0:
        # self.writer.add_scalar('action loss',action_loss,int(self.iter))
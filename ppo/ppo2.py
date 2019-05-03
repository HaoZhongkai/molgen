import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPO():
    def __init__(self, opt, policy, env, param):
        self.env = env
        self.policy = policy
        self.cut_out_tra = opt.cut_out_tra

        self.lr = param['lr']
        self.reward_decay = param['reward_decay']
        self.clip = param['clip']
        self.optimizer = opt.optimizer(self.policy.parameters(), lr=self.lr)



    ''' obs: (N,A) Tensor
        actions: action Tensor
        neglogpacs: log_probs Tensor
        G_t:'''
    def interaction(self):
        temp_obs, temp_actions, temp_values, temp_neglogpacs, temp_reward, temp_Gt = [],[],[],[],[],[]
        N,A,infos,_ = self.env.inits()
        info = {'reward': 0, 'property': 0, 'smiles': ''}
        terminal = False

        while not terminal:
            N = torch.Tensor(N).unsqueeze(0).long()
            A = torch.Tensor(A).unsqueeze(0)
            with torch.no_grad():
                action_prob, action, value = self.policy(N, A)

            log_prob = [Categorical(prob).log_prob(act) for prob,act in zip(action_prob,action)]
            action_np = [act.numpy()[0] for act in action]



            N_next,A_next,infos,terminal = self.env.step(action_np)
            info['reward'] += infos['reward_step']

            temp_obs.append((N,A))
            temp_actions.append(action)
            temp_values.append(value)
            temp_neglogpacs.append(log_prob)
            temp_reward.append(infos['reward_step'])

            N,A = N_next,A_next

        info['property'] = infos['qed']
        info['smiles'] = infos['smiles']
        '''截断trajectory'''
        if self.cut_out_tra:
            index = infos['best_index']
            temp_obs = temp_obs[:index]
            temp_actions = temp_actions[:index]
            temp_values = temp_values[:index]
            temp_neglogpacs = temp_neglogpacs[:index]
            temp_reward = temp_reward[:index]

        '''计算value'''
        R = 0
        for i in reversed(temp_reward):
            R = R * self.reward_decay + i
            temp_Gt.insert(0,R)



        return temp_obs, temp_actions, temp_values, temp_neglogpacs, temp_Gt, info

    def update(self, input):
        obs_, actions_, values_, neglogpacs_, Gt_ = input
        N, A = obs_
        actions_ = torch.Tensor(actions_)

        advantages = Gt_ - values_
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.squeeze()

        vpred, action_log_probs, dist_entropy = self.policy.evaluate_actions(N, A, actions_)

        # data reformat
        a_f_p, a_s_p, a_b_p, a_t_p = action_log_probs
        old_f_p, old_s_p, old_b_p, old_t_p = neglogpacs_

        ratio = torch.exp(a_f_p + a_s_p + a_b_p - old_f_p - old_s_p - old_b_p)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0
        # value_loss = (returns - vpred).pow(2).mean()

        self.optimizer.zero_grad()
        loss = value_loss + action_loss - dist_entropy.mean() * 1e-3
        loss.backward()
        # nn.utils.clip_grad_norm(self.net.parameters(), 0.5)
        self.optimizer.step()
        return

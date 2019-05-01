import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, input_num, output_num):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_num,64)
        self.fc2 = nn.Linear(64, output_num)

        self.fc3 = nn.Linear(input_num,64)
        self.fc4 = nn.Linear(64,1)

    def evaluate_actions(self, obs, actions):
        value, actions_prob = self.forward(obs)
        actions = actions.view(-1,)
        c = Categorical(actions_prob)
        actions_log_prob = c.log_prob(actions)
        entropy = c.entropy()
        actions_log_prob = actions_log_prob.view(-1,1)
        entropy = entropy.view(-1,1)
        return value, actions_log_prob, entropy


    def forward(self, x):
        value = F.relu(self.fc3(x))
        value = self.fc4(value)

        action_prob = F.relu(self.fc1(x))
        action_prob = torch.sigmoid(self.fc2(action_prob))

        return  value, action_prob

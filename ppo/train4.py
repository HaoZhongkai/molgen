from config import Config
from env.pg_env3 import *
import time
from ppo.ppo2 import PPO
from ppo.memory_buffer import MemoryBuffer
from gcn.policy_net import PolicyNet
from torchnet import meter
config = Config()
buffer = MemoryBuffer(100000,clean_ratio=0.3)

env = Mole_Env(config)
policynet = PolicyNet(config)
ppo = PPO(config,policynet,env)
clock = time.time()
reward = meter.AverageValueMeter()
qed = meter.AverageValueMeter()
max_qed = 0
smiles = []
iterate_ = 0
sample_num = 10
learn_num = 10
batch_size = 50

while True:
    for i in range(sample_num):
        obs, actions, values, neglogpacs, Gt, info = ppo.interaction()
        buffer.renew(obs,actions,values,neglogpacs,Gt)

        reward.add(info['reward'])
        qed.add(info['qed'])
        max_qed = max(max_qed, info['qed'])

    obs_, actions_, values_, neglogpacs_, Gt_ = buffer.sample(batch_size)


    for i in range(learn_num):

        returns = torch.Tensor(Gt_).view(-1, 1)
        values = torch.Tensor(values_).view(-1, 1)
        N = torch.Tensor([n[0] for n in obs_]).long()
        A = torch.Tensor([n[1] for n in obs_])

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.view(-1, 1)

        vpred, action_log_probs, dist_entropy = policynet.evaluate_actions(N, A, actions_)




        # ppo.update(obs_, actions_, values_, neglogpacs_)




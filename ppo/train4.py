from config import Config
from env.pg_env3 import *
import time
from ppo.ppo2 import PPO as PPO
from ppo.memory_buffer import MemoryBuffer
from gcn.policy_net import PolicyNet
from torchnet import meter

'''could run on single cpu'''

config = Config()
writer = config.writer
buffer = MemoryBuffer(100000,clean_ratio=0.3)

env = Mole_Env(config)
policynet = PolicyNet(config)
ppo_param = {'lr': 1e-3, 'reward_decay': 0.7, 'clip': 0.2}
ppo = PPO(config, policynet, env, ppo_param)


max_qed = 0
smiles = []
iterate_ = 0
sample_num = 10
learn_num = 10
batch_size = 50
value_loss_on = False

clock = time.time()
reward = meter.AverageValueMeter()
qed = meter.AverageValueMeter()

while True:

    for i in range(sample_num):
        obs, actions, values, neglogpacs, Gt, info = ppo.interaction()
        buffer.renew(obs,actions,values,neglogpacs,Gt)

        reward.add(info['reward'])
        # qed.add(info['qed'])
        # max_qed = max(max_qed, info['qed'])



    for i in range(learn_num):
        ppo.update(buffer.sample(batch_size))

    writer.add_scalar('reward', reward.value()[0])
    reward.reset()
    qed.reset()

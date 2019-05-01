import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter as Summary
from config import Config
from torch.autograd import Variable
from env.pg_env3 import *
from ppo import *
import time
from torchnet import meter
config = Config()
env = Mole_Env(config)
ppo = PPO(env,config)
ppo.net.load_state_dict(torch.load("/home/jeffzhu/MCTs/dataset/models_/0416_07_04.pkl"))
clock = time.time()

# load_path ='/home/jeffzhu/MCTs/dataset/models_/0405_20_29_1epoch.pkl'
# ppo.net.load_state_dict(torch.load(load_path))


smiles_path = config.LOGS_PATH + '/smiles.txt'
fp = open(smiles_path,'w+')

batch_size = 50
ppo_iteration = 20
writer = config.writer
# dummy_input = (torch.ones([1,13]).long(),torch.zeros([1,3,13,13]))
# with writer:
#     writer.add_graph(ppo.net,dummy_input)

reward = meter.AverageValueMeter()
qed = meter.AverageValueMeter()
max_qed = 0
smiles = []
iterate_ = 0


while True:
    obs, actions, values, neglogpacs, Gt, info = ppo.interaction()
    reward.add(info['reward'])
    qed.add(info['qed'])
    max_qed = max(max_qed,info['qed'])

    # for debug only
    smiles.append(info['smiles']+'\n')

    if len(ppo.actions) > 100:
        # index = np.random.choice(len(ppo.actions), batch_size, replace=False)
        obs_, actions_, values_, neglogpacs_, Gt_ = ppo.sample(batch_size)
        # for i in index:
        #     obs_.append(ppo.obs[i])
        #     actions_.append(ppo.actions[i])
        #     values_.append(ppo.values[i])
        #     neglogpacs_.append(ppo.neglogpacs[i])
        #     Gt_.append(ppo.Gt[i])
        iterate_ += 1
        '''可视化部分'''
        if iterate_%30==0:
            for item in ppo.net.named_parameters():
                if item[0] == 'GCN.atom_embed.weight':
                    pass
                    #writer.add_histogram('GCN3_grad',item[1].grad,int(iterate_/30))
                    #writer.add_scalar('max_grad_gcn3',torch.max(item[1].grad),int(iterate_/30))

            # fp.writelines(smiles)

        writer.add_scalar('reward',reward.value()[0])
        writer.add_scalar('average_qed',qed.value()[0])
        writer.add_scalar('max_qed',max_qed)

        print('REWARD:',reward.value()[0])
        reward.reset()
        qed.reset()

        # ppo.clean_buffer()

        if int((time.time()-clock)/(2*3600))>1:
            ppo.net.save()
            clock = time.time()


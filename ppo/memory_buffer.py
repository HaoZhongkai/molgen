
import random
import torch


class MemoryBuffer():
    def __init__(self,len,clean_ratio=0.3):
        self.buffer = []
        self.max_len = len
        self.current_size = 0
        self.clean_ratio = clean_ratio

    def renew(self,temp_obs,temp_actions,temp_values,temp_neglogpacs,temp_Gt):
        if self.current_size+len(temp_Gt)>self.max_len:
            self.clean()
        for i in range(len(temp_Gt)):
            self.buffer.append((temp_obs[i],temp_actions[i],temp_values[i],temp_neglogpacs[i],temp_Gt[i]))
        self.current_size = len(self.buffer)
        return

    '''output:
        obs : list of (N,A) tensors with batch
        actions :  tensor , tensor,tensor with batch
    '''
    def sample(self,batch_size):
        output = random.sample(self.buffer, batch_size)
        # data reformate
        obs, actions, values, neglogpacs, Gt = [], [], [], [], []

        obs = [output[i][0] for i in range(batch_size)]
        obs = torch.cat([n[0] for n in obs], dim=0), torch.cat([n[1] for n in obs], dim=0)
        actions = torch.Tensor([output[i][1] for i in range(batch_size)])
        values = torch.Tensor([output[i][2] for i in range(batch_size)])
        neglogpacs = [torch.cat([output[i][3][j] for i in range(batch_size)]) for j in range(4)]
        Gt = torch.Tensor([output[i][4] for i in range(batch_size)])

        # for i in range(len(output)):
        #     obs.append(output[i][0])
        #     actions.append(output[i][1])
        #     values.append(output[i][2])
        #     neglogpacs.append(output[i][3])
        #     Gt.append(output[i][4])

        return (obs, actions, values, neglogpacs, Gt)


    def clean(self):
        random.shuffle(self.buffer)
        self.buffer = self.buffer[:int(self.clean_ratio*self.current_size)]
        return
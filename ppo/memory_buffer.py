
import random



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


    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)


    def clean(self):
        random.shuffle(self.buffer)
        self.buffer = self.buffer[:int(self.clean_ratio*self.current_size)]
        return
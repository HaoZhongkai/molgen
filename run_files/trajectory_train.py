from gcn.policy_net import PolicyNet
from torch.utils.data import DataLoader
from config import Config
from torchnet import meter
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from ppo.dataset_pre_train import Tra_Dataset


torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Trainer():
    def __init__(self,model=None,opt=Config()):
        self.model = model
        self.opt = opt
        self.criterion = opt.criterion

        self.optimizer = opt.optimizer(self.model.parameters(),lr = opt.lr)
        self.log_path = opt.LOGS_PATH
        self.writer = SummaryWriter(log_dir=self.opt.LOGS_PATH)
        if opt.use_gpu:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)



    def train(self,train_data,val_data=None):
        print('Now Begin Training!')
        train_loader = DataLoader(train_data,batch_size=self.opt.batch_size,shuffle=True)


        if self.opt.use_gpu:
            self.model.cuda()

        #meter initialize
        CM_meter = meter.ConfusionMeter(13)
        loss_meter = meter.AverageValueMeter()
        previous_loss = 1e10

        for epoch in range(self.opt.max_epoch):
            loss_meter.reset()
            CM_meter.reset()
            #train
            for i,(H, A, a_first, a_second, a_edge, a_stop) in enumerate(train_loader):
                # a_first, a_second, a_edge, a_stop = label
                if self.opt.use_gpu:
                    H = H.cuda()
                    A = A.cuda()
                    a_first, a_second, a_edge, a_stop = a_first.cuda(),a_second.cuda(),a_edge.cuda(),a_stop.cuda()

                self.optimizer.zero_grad()
                score,action,_ = self.model(H,A)


                loss_first,loss_second,loss_edge,loss_stop = (self.criterion(score[0],a_first),
                                                              self.criterion(score[1],a_second),
                                                              self.criterion(score[2],a_edge),
                                                              self.criterion(score[3],a_stop))
                loss = loss_first+loss_second+loss_edge+loss_stop
                loss.backward()
                self.optimizer.step()



                # abs_loss.add(torch.abs(score-label).sum().detach().cpu()/self.opt.batch_size)
                loss_meter.add(loss.data.cpu())
                CM_meter.add(action[1],a_second.squeeze())
                accuracy = 100*sum(CM_meter.value()[i,i] for i in range(13))/CM_meter.value().sum()

                if i % self.opt.print_feq == self.opt.print_feq - 1:
                    nither = epoch*len(train_loader)+i
                    print('EPOCH:{0},i:{1},loss:{2},accuracy:{3}'.format(epoch,i,loss_meter.value()[0],accuracy))
                    self.writer.add_scalar('train_loss',loss_meter.value()[0],nither)


                #tensorboard visulize module


            # if val_data:
            #     val_loss = self.test(val_data,val=True)
            #     print('val loss:',val_loss)
            #     self.writer.add_scalar('val_loss',val_loss,epoch)

            print('!!!!!!now{0},previous{1}'.format(loss_meter.value()[0],previous_loss))
            if loss_meter.value()[0]>=previous_loss:
                self.opt.lr = self.opt.lr*self.opt.lr_decay
                print('!!!!!LR:',self.opt.lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.opt.lr

            previous_loss = loss_meter.value()[0]

'''
    def test(self,test_data,val=False):

        if self.opt.use_gpu:
            self.model.cuda()

        self.model.eval()
        test_loader = DataLoader(test_data,batch_size=self.opt.batch_size,shuffle=True)
        result = []
        loss_meter = meter.AverageValueMeter()
        for i,(H, A, label) in enumerate(test_loader):

            #数据格式转换
            if self.opt.use_gpu:
                H = H.cuda()
                A = A.cuda()
                label = label.cuda()
                label = torch.unsqueeze(label, 1)  # 数据预处理问题补丁

            score = self.model(H, A)
            loss = self.criterion(score,label)
            loss_meter.add(loss.data.cpu().detach().numpy())


            if not val:
                result.append(score.cpu().detach().numpy())



        self.model.train()
        if val:
            return loss_meter.value()[0]
        else:
            result = np.stack(result)
            return result,loss_meter.value()[0]'''


def main():
    dconfig = Config()
    dconfig.batch_size = 50
    dconfig.criterion = torch.nn.CrossEntropyLoss()
    dconfig.optimizer = torch.optim.SGD
    dconfig.lr = 5e-3
    dconfig.max_epoch = 5
    file_path = '/home/jeffzhu/MCTs/dataset/datasets/rand_sample1.pkl'
    net = PolicyNet(dconfig)
    # net.load_state_dict(torch.load('/home/jeffzhu/MCTs/dataset/models_/0405_20_21.pkl'))
    trainer = Trainer(net,dconfig)
    tra_dataset = Tra_Dataset(file_path,200000)
    trainer.train(tra_dataset)
    net.save()



if __name__ == '__main__':
    main()
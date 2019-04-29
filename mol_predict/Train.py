from torch.utils.data import DataLoader
from config import Config
from torchnet import meter
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter


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
        loss_meter = meter.AverageValueMeter()
        abs_loss = meter.AverageValueMeter()
        previous_loss = 1e10

        for epoch in range(self.opt.max_epoch):
            loss_meter.reset()
            abs_loss.reset()

            #train
            for i,(H, A, label) in enumerate(train_loader):
                if self.opt.use_gpu:
                    H = H.type(torch.long).cuda()
                    A = A.cuda()
                    label = label.cuda()
                    label = torch.unsqueeze(label,1)        #数据预处理问题补丁

                self.optimizer.zero_grad()
                score = self.model(H,A)
                # print(i)
                loss = self.criterion(score,label)
                loss.backward()
                self.optimizer.step()



                abs_loss.add(torch.abs(score-label).sum().detach().cpu()/self.opt.batch_size)
                loss_meter.add(loss.data.cpu())

                if i % self.opt.print_feq == self.opt.print_feq - 1:
                    nither = epoch*len(train_loader)+i
                    print('EPOCH:{0},i:{1},loss:{2},abs_loss:{3}'.format(epoch,i,loss.data.cpu(),abs_loss.value()[0]))
                    self.writer.add_scalar('train_loss',loss_meter.value()[0],nither)
                    self.writer.add_scalar('abs_loss',abs_loss.value()[0],nither)

                #tensorboard visulize module


            if val_data:
                val_loss = self.test(val_data,val=True)
                print('val loss:',val_loss)
                self.writer.add_scalar('val_loss',val_loss,epoch)

            print('!!!!!!now{0},previous{1}'.format(loss_meter.value()[0],previous_loss))
            if loss_meter.value()[0]>=previous_loss:
                self.opt.lr = self.opt.lr*self.opt.lr_decay
                print('!!!!!LR:',self.opt.lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.opt.lr

            previous_loss = loss_meter.value()[0]


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
                H = H.type(torch.long).cuda()
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
            return result,loss_meter.value()[0]







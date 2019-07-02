import torch
from mol_predict.data_util import QM9qed_dataset,QM9U0_dataset
from mol_predict.Train import Trainer
from mol_predict.rGCN import RGCN as GCN_QED
from config import Config


torch.set_default_tensor_type(torch.cuda.FloatTensor)
dconfig = Config()

load_model = True
# file_path = dconfig.DATASET_PATH+'/gdb9_qed.pkl'
# file_path = dconfig.DATASET_PATH+'/gdb9_U0.pkl'
file_path = ''
# load_path = '/home/jeffzhu/MCTs/dataset/models_/0527_22_54.pkl'
load_path = None

train_dataset,valid_dataset,test_dataset = QM9qed_dataset(file_path,valid=True).Get_data()

GCN_qed = GCN_QED(dconfig)
print(GCN_qed)
if load_model:
    GCN_qed.load_state_dict(torch.load(load_path))

trainer = Trainer(model=GCN_qed,opt=dconfig)

trainer.train(train_dataset,valid_dataset)

_,tloss = trainer.test(test_dataset,val=False)

print('test loss:',tloss)
GCN_qed.save()
print('save success!')

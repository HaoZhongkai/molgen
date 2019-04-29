import torch
from mol_predict.data_util import QM9qed_dataset,QM9U0_dataset
from mol_predict.Train import Trainer
from mol_predict.gcn_qed2 import GCN_QED
from config import Config


torch.set_default_tensor_type(torch.cuda.FloatTensor)
dconfig = Config()
file_path = dconfig.DATASET_PATH+'/gdb9_qed.pkl'
# file_path = dconfig.DATASET_PATH+'/gdb9_U0.pkl'
load_path = '/home/jeffzhu/MCTs/dataset/models_/0327_17_07.pkl'

train_dataset,valid_dataset,test_dataset = QM9qed_dataset(file_path,valid=True).Get_data()

GCN_qed = GCN_QED(dconfig)
GCN_qed.load_state_dict(torch.load(load_path))

trainer = Trainer(model=GCN_qed,opt=dconfig)

trainer.train(train_dataset,valid_dataset)

_,tloss = trainer.test(test_dataset,val=False)

print('test loss:',tloss)
GCN_qed.save()
print('save success!')


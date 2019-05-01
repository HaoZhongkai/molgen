from tensorboardX import SummaryWriter
import torch
from Mol_predict.Data_util import QM9qed_dataset
from config import config
import pickle
dconfig = config()
file_path = dconfig.DATASET_PATH+'/gdb9_qed.pkl'

writer = SummaryWriter(log_dir=dconfig.LOGS_PATH)
data = pickle.load(open(file_path,'rb'))
train_qed = data[0][2]
# train_dataset,valid_dataset,test_dataset = QM9qed_dataset(file_path,valid=True).Get_data()


writer.add_histogram('qed',train_qed)
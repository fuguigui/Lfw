import torch
import torch.utils.data as data
from torch.autograd import Variable
import utils.training as train_utils

import utils.lfw as lfw

# load the datasets
print("Loading the data......")
train_dt = lfw.Lfw("./datasets/","train_expr.txt",'/home/guigui/final_proj')

# batch the datasets
print("Batching the datasets......")
batch_size = 2
train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=False)
print("Train:%d"%len(train_loader.dataset.imgs))

for idx, data in enumerate(train_loader):
    inputs = Variable(data[0])
    targets = Variable(data[1])
    train_utils.save_results(targets, 1, 0.9,0.9)


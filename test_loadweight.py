import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import utils.training as train_utils
import model.ThiNet as ThiNet

import model.mini_fcn as fcn
import utils.lfw as lfw

# load the datasets
print("Loading the data......")
train_dt = lfw.Lfw("./datasets/","train.txt",'/home/fugr/graduate/final_proj')
valid_dt = lfw.Lfw("./datasets/","validation.txt",'/home/fugr/graduate/final_proj')


# batch the datasets
print("Batching the datasets......")
batch_size = 20
train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)
print("Train:%d"%len(train_loader.dataset.imgs))
valid_loader = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=True)
print("Valid:%d"%len(valid_loader.dataset.imgs))


# Build the nets
print("Building the nets...")
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
COMPREES_RATE = 0.8
torch.manual_seed(0)

fcn_model = fcn.fcn8s()

optimizer = optim.RMSprop(fcn_model.parameters(), lr=LR)
criterion = nn.MSELoss()
train_helper = train_utils.trainHelper(fcn_model, optimizer, criterion,LR_DECAY= LR_DECAY, n_epochs=2)
train_helper.load_weights("./weights/06-13-13-39/weights-7-0.056-0.103.pth")
fcn_model = train_helper.getmodel()
optimizer = optim.RMSprop(fcn_model.parameters(),lr=LR)
train_helper.setOptimizer(optimizer)

train_helper.FullExpr(train_loader, valid_loader, train_dt, if_classes=True, n_classes=3)


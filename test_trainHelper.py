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
train_dt = lfw.Lfw("./datasets/","train_expr.txt",'/home/guigui/final_proj')
valid_dt = lfw.Lfw("./datasets/","valid_expr.txt",'/home/guigui/final_proj')
# test_dt = lfw.Lfw("./datasets/","test.txt",'/home/guigui/final_proj')


# batch the datasets
print("Batching the datasets......")
batch_size = 4
train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=False)
print("Train:%d"%len(train_loader.dataset.imgs))
valid_loader = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False)
print("Valid:%d"%len(valid_loader.dataset.imgs))
# test_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size, shuffle=False)
# print("Test :%d"%len(test_loader.dataset.imgs))


# Build the nets
print("Building the nets...")
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
COMPREES_RATE = 0.8
torch.manual_seed(0)

fcn_model = fcn.fcn32s()
fcn_model.apply(train_utils.weights_init)

optimizer = optim.RMSprop(fcn_model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()
train_helper = train_utils.trainHelper(fcn_model, optimizer, criterion, n_epochs=8)

print(fcn_model.parameters())

train_helper.FullExpr(train_loader, valid_loader, lfw, if_classes=True)

# Test ThiNet.
fcn_thinnet = ThiNet(fcn_model)
fcn_thinnet.thinmodel(train_dt)

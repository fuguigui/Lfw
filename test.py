import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import time
from model import fcn
import utils.lfw as lfw
import utils.training as train_utils

# load the datasets
print("Loading the data......")
train_dt = lfw.Lfw("./datasets/","train_expr.txt",'/home/guigui/final_proj')
valid_dt = lfw.Lfw("./datasets/","validation.txt",'/home/guigui/final_proj')
test_dt = lfw.Lfw("./datasets/","test.txt",'/home/guigui/final_proj')


# batch the datasets
print("Batching the datasets......")
batch_size = 2
train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=False)
print("Train:%d"%len(train_loader.dataset.imgs))
valid_loader = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False)
#print("Valid:%d"%len(valid_loader.dataset.imgs))
test_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size, shuffle=False)
#print("Test :%d"%len(test_loader.dataset.imgs))


# Build the nets
print("Building the nets...")
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 2
torch.manual_seed(0)
valid_loss_best = 0
valid_err_best = 0

fcn_model = fcn.fcn32s(n_classes=3)
print(fcn_model)
fcn_model.apply(train_utils.weights_init)
# ????? what is model.parameters()?
print(fcn_model.parameters())
optimizer = optim.RMSprop(fcn_model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()

# Train the model
for epoch in range(1, N_EPOCHS + 1):
    since = time.time()

    ### Train ###
    trn_loss, trn_err = train_utils.train(
        fcn_model, train_loader, optimizer, criterion)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1 - trn_err))
    time_elapsed = time.time() - since
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Valid ###
    valid_loss, valid_err = train_utils.test(fcn_model, valid_loader, criterion, epoch)
    print('Valid - Loss: {:.4f} | Acc: {:.4f}'.format(valid_loss, 1 - valid_err))
    time_elapsed = time.time() - since
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Checkpoint ###
    if valid_loss_best > 0:
        if valid_loss < valid_loss_best and valid_err < valid_err_best:
            valid_loss_best = valid_loss
            valid_err_best = valid_err
            train_utils.save_weights(fcn_model, epoch, valid_loss, valid_err)
    else:
        valid_loss_best = valid_loss
        valid_err_best = valid_err

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer,
                                     epoch, DECAY_EVERY_N_EPOCHS)
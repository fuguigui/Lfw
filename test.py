import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np
from model import FCDenseNet
import utils.lfw as lfw
import utils.training as train_utils

# load the datasets
train_dt = lfw.Lfw("./datasets/","train.txt", transform=transforms.ToTensor(), loader=lfw.my_loader)
valid_dt = lfw.Lfw("./datasets/","validation.txt", transform=transforms.ToTensor(), loader=lfw.my_loader)
test_dt = lfw.Lfw("./datasets/","test.txt", transform=transforms.ToTensor(), loader=lfw.my_loader)

# batch the datasets
batch_size = 4
train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=False)
print("Train:%d"%len(train_loader.dataset.imgs))
valid_loader = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False)
print("Valid:%d"%len(valid_loader.dataset.imgs))
test_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size, shuffle=False)
print("Test :%d"%len(test_loader.dataset.imgs))


for i, (batch_x, batch_y) in enumerate(train_loader):
    if(i<4):
        print(i,batch_x.size(), batch_y.size())
        #lfw.show_batch(batch_x)
        # show_batch(batch_y)
        #plt.axis('off')
        #plt.show()

# Build the nets
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 2
torch.manual_seed(0)

model = FCDenseNet.FCDenseNet67(n_classes=3)
print(model)
model.apply(train_utils.weights_init)
# ????? what is model.parameters()?
optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(1, N_EPOCHS + 1):
    since = time.time()

    ### Train ###
    trn_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1 - trn_err))
    time_elapsed = time.time() - since
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Test ###
    test_loss, test_err = train_utils.test(model, test_loader, criterion, epoch)
    print('Test - Loss: {:.4f} | Acc: {:.4f}'.format(test_loss, 1 - test_err))
    time_elapsed = time.time() - since
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Checkpoint ###
    train_utils.save_weights(model, epoch, test_loss, test_err)

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer,
                                     epoch, DECAY_EVERY_N_EPOCHS)
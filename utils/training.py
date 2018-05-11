import os
import sys
import math
import string
import random
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

import utils.lfw as img_utils

RESULTS_PATH = './results/'
WEIGHTS_PATH = './weights/'


def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def save_results(output, epoch ,loss, err):
    results_folder = 'results-%d-%.3f-%.3f' % (epoch, loss, err)
    results_fpath = os.path.join(RESULTS_PATH, results_folder)
    for idx, item in enumerate(output):
        img_utils.save_output(results_fpath,idx, item)

    print("Prediction results are saved!")


def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return round(err,5)

def train(model, trn_loader, optimizer, criterion):
    model.train()
    trn_loss = 0
    trn_error = 0

    length = len(trn_loader)
    print("lenght of train loader is ",length)

    for idx, data in enumerate(trn_loader):
        inputs = Variable(data[0])
        targets = Variable(data[1])

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.data[0]
        pred = get_predictions(output)
        right = get_predictions(targets)
        trn_error += error(pred, right)

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return output, trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        # ????? What is Variable()?
        data = Variable(data, volatile=True)
        target = Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = get_predictions(output)
        right = get_predictions(target)
        test_error += error(pred, right)
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        #img_utils.view_annotated(targets[i])

def my_loss(outputs, targets):
    return (outputs-targets)**2;
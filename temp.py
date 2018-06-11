from model import fcn
import utils.training as train_utils
from model.ThiNet import ThiNet
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
import scipy.misc as m
import numpy as np
import os
import utils.lfw as lfw
from skimage import io
import torch.nn.init as init
import torch
from torch.autograd import Variable

#
# fcn_model = fcn.fcn32s(n_classes=3)
# new_model = ThiNet(fcn_model)
#
# print("original model:")
# print(fcn_model)
# children = list(fcn_model.named_children())
# modules = list(fcn_model._modules)
# name2, submodel = children[1]
# new_model._modules[name2] = nn.Sequential(
#             OrderedDict([
#                 ("conv1",nn.Conv2d(3,64,1,padding=4)),
#                 ("relu1",nn.ReLU(inplace=True)),
#                 ("conv2",nn.Conv2d(2,2,1,padding=1)),
#                 ("relu2",nn.ReLU(inplace=True)),
#                 ("max",nn.MaxPool2d(2,stride=2,ceil_mode=True))
#             ]))
# new_model.thinmodel()
# new_model.filter_selection(30,)
# print("Changed model:")
# print(new_model.get_model())
#
# for name, submodel in fcn_model.named_children():
#     print(new_model._modules[name])
#
#
# fcn_model.apply(train_utils.weights_init)
#
# params = fcn_model.state_dict()
# for k,v in params.items():
#     print(k)
#     bias = k.replace("weight","bias")
#     print(params[k])
#     print(params[bias])
#
# print(fcn_model.parameters())
# layer = nn.Conv2d(2,4,5,padding=1)
# print('The parameters of layer:')
# print('out channels:', layer.out_channels)
# print('in channels:',layer.in_channels)
# print('test over!')


# ---------------Test: tensor operations
# x = torch.FloatTensor(2,3,4)
# y = x.narrow(0,0,1)
#
# print(y)
# y[0]=x[1]
# print(y)

# ---------------Test: ThiNet.CalculSqX:
# inputs= np.array([[[[1,0,2],[0,0.6,0]],
#                    [[1,1,2],[0,1,0]],
#                    [[0,1,0],[1,1,1]]],
#                   [[[0,0,2],[0,1,0]],
#                    [[0,0,2],[0,1,0]],
#                    [[1,0,0],[1,1,1]]],
#                   [[[2,0,2],[0,1,0]],
#                    [[1,0,2],[0,1,1]],
#                    [[1,0,2],[0,1,2]]],
#                   [[[1, 0, 2], [2, 1, 0]],
#                    [[1, 0, 2], [1, 1, 0]],
#                    [[0, 0, 0], [0, 1, 1]]],
#                   [[[1, 0, 2], [1, 1, 0]],
#                    [[1, 0, 2], [2, 1, 0]],
#                    [[0, 0, 0], [1, 0.2, 1]]]])
# set=[0]
# def CalculSqX(inputs, set):
#     sum = 0.0
#     n, c, h, w = inputs.shape
#     for i in range(n):
#         sub_sum = 0.0
#         for j in range(c):
#             if j in set:
#                 for k in range(h):
#                     for l in range(w):
#                         sub_sum +=inputs[i][j][k][l]
#         sum += sub_sum*sub_sum
#     return sum
#
# sum = CalculSqX(inputs,set)
# print(sum)


# ---------------Test: ThiNet.filter_selection:
# def filter_selection(channel_num, input):
#     T = []
#     I = list(range(channel_num))
#     while len(T) < channel_num * 0.3:
#         min_value = 1000
#         min_id = -1
#         for i in I:
#             tempT = T.copy()
#             tempT.append(i)
#             sum = CalculSqX(input, tempT)
#             if sum < min_value:
#                 min_value = sum
#                 min_id = i
#         if min_id != -1:
#             T.append(min_id)
#             I.remove(min_id)
#     return T
# n,c,h,w = inputs.shape
# reduced_T = filter_selection(c, inputs)
# print("Reduced result:",reduced_T)
#

# -----------------Test: ThiNet.drop_filter(cur, extra_filters):
# def drop_filter(cur, extra_filters):
#     # TODO: change sudocode into Python
#     # Only applicable to Conv2d???
#     out_num = cur.out_channels
#     in_num = cur.in_channels
#     kernel = cur.kernel_size
#     pad = cur.padding
#
#     new_out = out_num - len(extra_filters)
#     new_layer = nn.Conv2d(in_num, new_out, kernel, padding=pad)
#
#     new_dict = new_layer.state_dict()
#     pretrained_dict = cur.state_dict()
#     item = pretrained_dict.items()
#     modified_dict = {}
#     for k,v in item:
#         new_v =v.narrow(0,0,new_out)
#         j = 0
#         for i in range(out_num):
#             if i in extra_filters:
#                 continue
#             new_v[j] = v[i]
#             j = j+1
#         modified_dict[k]=new_v
#     new_dict.update(modified_dict)
#     new_layer.load_state_dict(new_dict)
#
#     return new_layer
# Test -------------drop_filters for next layer
# def drop_filter(next, extra_filters):
#     if (not isinstance(next, nn.Conv2d)):
#         print("The layer is not Conv2d!")
#         return
#     next_out_num = next.out_channels
#     next_in_num = next.in_channels
#     next_kernel = next.kernel_size
#     next_pad = next.padding
#
#     new_in = next_in_num - len(extra_filters)
#     next_new_layer = nn.Conv2d(new_in, next_out_num, next_kernel, padding=next_pad)
#     next_new_dict = next_new_layer.state_dict()
#     next_pretrained_dict = next.state_dict()
#     next_modified_dict = {}
#     v = next_pretrained_dict['weight']
#     new_v = v.narrow(1, 0, new_in)
#     j = 0
#     for i in range(next_in_num):
#         if i in extra_filters:
#             continue
#         for l in range(next_out_num):
#             new_v[l][j] = v[l][i]
#         j = j + 1
#     next_modified_dict['weight'] = new_v
#     v = next_pretrained_dict['bias']
#     next_modified_dict['bias']=v
#
#     next_new_dict.update(next_modified_dict)
#     next_new_layer.load_state_dict(next_new_dict)
#     return next_new_layer

# cur_layer = nn.Conv2d(3,10,2,padding=1)
# init.xavier_uniform(cur_layer.weight,gain=1)
# init.constant(cur_layer.bias,0.1)
#
# max_layer = nn.ReLU(inplace=True)
# new_max_layer_in_model = new_model.drop_filter(max_layer,reduced_T)
# new_conv_layer_in_model = new_model.drop_filter(cur_layer,reduced_T)
# print(new_layer)
# cur_layer = nn.Conv2d(10,4,2,padding=1)
# init.xavier_uniform(cur_layer.weight, gain = 1)
# init.constant(cur_layer.bias,0.005)
# new_conv_layer = drop_filter(cur_layer,reduced_T)
# print(cur_layer.state_dict())
# print(new_conv_layer.state_dict())

# Test: ThiNet.drop_filter
# cur_layer = nn.Conv2d(3,10,2,padding=1)
# next_layer = nn.Conv2d(10,6,4,padding=1)
# new_cur_layer, new_next_layer = new_model.drop_filter(cur_layer,next_layer,reduced_T)
# print(cur_layer.state_dict())
# print(new_cur_layer.state_dict())
#
# print("Next layer.parameters")
# print(next_layer.state_dict())
# print(new_next_layer.state_dict())

# Test ----Variable .shape
# inputs = torch.from_numpy(inputs)
# inputs = Variable(inputs)
# numpy_input = inputs.data.numpy()
# print(numpy_input.shape)

# # Test -- thinmodel
# new_model.thinmodel(inputs)
# new_model.save_model()

# Test -- nn.Conv2d
# inputs = Variable(torch.from_numpy(inputs))
# layer = nn.Conv2d(3,64,2,padding=1)
# init.xavier_uniform(layer.weight, gain=1)
# init.constant(layer.bias,0.1)
# layer.double()
# # output= layer(inputs)
# output = layer.forward(inputs)
# print(output)

# Test ------ calculate each_class accuracy
# def get_predictions(output_batch):
#     bs,c,h,w = output_batch.size()
#     values, indices = output_batch.cpu().max(1)
#     indices = indices.view(bs,h,w)
#     return indices
#
# def each_class_acc(preds, targets, classes):
#     bs,h,w = preds.size()
#
#     n_pixels = bs*h*w
#     preds.resize_(n_pixels)
#     targets.resize_(n_pixels)
#
#     same_list = []
#     for idx in range(n_pixels):
#         if (preds[idx] == targets[idx]):
#             same_list.append(preds[idx])
#
#     acc_class = []
#     for i in range(classes):
#         total = 0
#         for j in range(n_pixels):
#             if (preds[j] == i):
#                 total+=1
#         sub = 0
#         for j in range(len(same_list)):
#             if (same_list[j] == i):
#                 sub+=1
#         acc_class.append(sub / total)
#     return acc_class
#
#
# targets= np.array([[[[1,0,0],[1,0.6,0]],
#                    [[1,3,2],[0,1,0]],
#                    [[1,0,0],[1,2,1]]],
#                   [[[0,0,2],[0,1,0]],
#                    [[0,0,2],[0,1,0]],
#                    [[1,0,0],[1,2,1]]],
#                   [[[2,0,2],[0,1,0]],
#                    [[1,0,2],[0,0,1]],
#                    [[1,0,0],[0,1,2]]],
#                   [[[1, 0, 2], [2, 1, 0]],
#                    [[1, 2, 0], [1, 0.5, 0]],
#                    [[0, 0, 0], [0, 0, 1]]],
#                   [[[1, 0, 2], [1, 1, 0]],
#                    [[1, 0, 2], [2, 1, 0]],
#                    [[1, 0, 0], [2, 0.2, 1]]]])
# targets = torch.from_numpy(targets)
#
# output_pred = train_utils.get_predictions(inputs)
# target_pred = train_utils.get_predictions(targets)
#
# res, acc_list = train_utils.error(output_pred, target_pred, 3)


# -----------------Test: lfw

train_dt = lfw.Lfw("./datasets/","train_expr.txt",'/home/guigui/final_proj')

img, lbl = train_dt[2]
#m.imsave('./results/img.ppm',img)
train_dt.save_output('./results/',2,lbl)

# ---------------Test: os.path.join

# RESULTS_PATH_1 = './results/'
# savepath = 'fugr'
# savepath2 = 'fugg/'
# results_fpath = 'bottle'
# RESULTS_PATH_2 = './results'
# results_path = os.path.join(RESULTS_PATH_1, savepath, results_fpath)
# print('The saving path is ', results_path)
#
# results_path2 = os.path.join(RESULTS_PATH_2, savepath2, results_fpath)
# print('The saving path is ', results_path2)
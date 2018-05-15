import torch.nn as nn
import torch
import time
from collections import OrderedDict

class ThiNet(object):
    def __init__(self, pretrained_model, if_file = False, compress_rate = 0.9):
        self.compress_rate = compress_rate
        if(if_file):
            self.model_name = pretrained_model
            self.model = torch.load(pretrained_model)
        else:
            self.model = pretrained_model
            self.model_name = "Direct model"
        self.time_str = time.strftime('%y-%m-%d-%H',time.localtime(time.time()))

    def prune_block(self, old_block, inputs):
        new_block = old_block
        modules = list(old_block.named_children())
        lay_num = len(modules)
        each_input = inputs
        for itr in range(lay_num-1):
            cur_layer_name, cur_layer = modules[itr]
            next_layer_name, next_layer = modules[itr+1]

            # use the current layer to get the output
            each_output = cur_layer(each_input)
            # new_layer = nn.Conv2d(3,9,3,padding=1)
            new_layer = self.prune_layer(cur_layer, next_layer, each_input, each_output)
            new_block._modules[cur_layer_name] = new_layer

            each_input = new_layer(each_input)
        return new_block

    def prune_layer(self, cur, next, input, output):
        # TODO: check this function
        next_input = cur(input)
        extra_filters = self.filter_selection(next, next_input,output)
        pruned_layer = self.drop_filter(cur, extra_filters)
        return self.fine_tune(pruned_layer)

    # def fine_tune(self, layer):
    # TODO: fine_tune: fine_tune the layer. Need more parameters? like input, output?
    # TODO: don't know how to do.  Implementation in line 31.


    def thinmodel(self, inputs):
        model = self.model
        children  = list(model.named_children())
        for name,submodel in children:
            model._modules[name] = self.prune_block(submodel,inputs)
            inputs = submodel(inputs)
        self.model = model

    # def defaultblock(self):
    #     layer= nn.Sequential(
    #         OrderedDict(
    #             [("conv1", nn.Conv2d(3, 9, 3, padding=2)),
    #              ("relu1", nn.ReLU(inplace=True)),
    #              ("max", nn.MaxPool2d(2, stride=2, ceil_mode=True))]))
    #     return layer
    def filter_selection(self, layer, input, output):
        # TODO: change sudocode into Python
        T=[]
        I = range(channel_num)
        while size(T)<channel_num*self.compress_rate:
            min_value = 1000
            min_id = -1
            for i in I:
                tempT = append(T, i)
                sum = CalculSqX(input, tempT)
                if sum < min_value:
                    min_value = sum
                    min_id = i
            if min_id!=-1:
                T.append(min_id)
                I.remove(min_id)
        return T

    def drop_filter(self, cur, extra_filters):
        # TODO: change sudocode into Python
        # Only applicable to Conv2d???
        in_num, out_num, channel, pad = cur.getParameters()
        new_out = out_num- len(extra_filters)
        new_layer = nn.Conv2d(in_num, new_out, channel, padding = pad)
        j = 0
        for i in range(out_num):
            if i in extra_filters:
                continue
            new_layer[j].parameters = cur[i].parameters()

        return new_layer
    def save_model(self):
        torch.save(self.model,self.time_str+self.model_name)
    def get_model(self):
        return self.model


def CalculSqX(inputs, set):
    # TODO: change sudocode into Python
    sum = 0.0
    n, m = dim(input)
    for i in range(n):
        sub_sum = 0.0
        for j in range(m):
            if j in set:
                sub_sum +=inputs[i][j]
        sum += sub_sum*sub_sum
    return sum




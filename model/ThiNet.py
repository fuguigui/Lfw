import torch.nn as nn
import torch
import time
import torch.optim as optim
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

    def prune_block(self, new_in_num, old_block, inputs):
        new_block = old_block
        modules = list(old_block.named_children())
        lay_num = len(modules)
        out_num = 0
        each_input = inputs
        for itr in range(lay_num-1):
            cur_layer_name, cur_layer = modules[itr]
            if(not isinstance(cur_layer, nn.Conv2d)):
                continue

            # Find the next conv2d layer in the block
            middle_layers=[]
            next_itr = itr+1
            while(next_itr< lay_num):
                next_layer_name, next_layer = modules[next_itr]
                if(isinstance(next_layer,nn.Conv2d)):
                    break
                middle_layers.append(next_layer)
                next_itr = next_itr+1
            if(next_itr<lay_num):
                itr = next_itr-1
            else:
                break

            # use the current layer to get the output
            each_output = cur_layer(each_input)
            out_num, new_cur_layer, new_next_layer = self.prune_layer(new_in_num, cur_layer, next_layer, each_input, each_output)
            new_block._modules[cur_layer_name] = new_cur_layer
            new_block._modules[next_layer_name] = new_next_layer

            each_input = new_cur_layer(each_input)

        return out_num, new_block

    def prune_layer(self, new_in_num, cur, next,middle_layers, input, output):
        next_input = cur(input)
        extra_filters = self.filter_selection(next.in_channels, next_input)
        cur_pruned_layer,next_pruned_layer = self.drop_filter(cur, next, new_in_num, extra_filters)
        return self.fine_tune(cur_pruned_layer, next_pruned_layer, middle_layers, input, output)

    def fine_tune(self, cur_layer, next_layer,middle_layers, input, target):
        model = nn.Module()
        model.add_module(cur_layer)
        for i in range(len(middle_layers)):
            model.add_module(middle_layers[i])
        model.add_module(next_layer)

        optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    #
    # def thinmodel(self, inputs):
    #     model = self.model
    #     children = list(model.named_children())
    #     new_in = 0
    #     for name,submodel in children:
    #         new_in, model._modules[name] = self.prune_block(new_in, submodel,inputs)
    #         inputs = submodel(inputs)
    #     self.model = model

    def thinmodel(self, inputs):
        model = self.model
        children = list(model.named_children())
        modules = model._modules
        all_layers=[]
        for name,module in modules.items():
            for layer_name, layer in module._modules.items():
                each_layer=[name,layer_name,layer]
                all_layers.append(each_layer)
        print(len(all_layers))

        layer_len = len(all_layers)
        new_in = 0
        pre = 0,next = 0
        each_input = inputs
        # Find the first layer to be pruned
        for i in range(layer_len):
            layer = all_layers[i][2]
            if(isinstance(layer,nn.Conv2d)):
                next = i
                break

        while(pre < layer_len):
            pre = next
            middle_layers=[]
            while (next<layer_len):
                layer = all_layers[next][2]
                if (isinstance(layer, nn.Conv2d)):
                    break
                else:
                    middle_layers.append(all_layers[next][2])
                    next = next+1
            if(next == layer_len):
                break
            each_output = all_layers[next][2](all_layers[pre][2](each_input))
            new_in, new_cur_layer, new_next_layer = \
                self.prune_layer(new_in, all_layers[pre][2], all_layers[next][2],
                                 middle_layers,each_input,each_output)
            # Save in the current model
            cur_module_name = all_layers[pre][0]
            cur_layer_name = all_layers[pre][1]
            next_module_name = all_layers[next][0]
            next_layer_name = all_layers[next][1]

            model._modules[cur_module_name]._modules[cur_layer_name] = new_cur_layer
            model._modules[next_module_name]._modules[next_layer_name] = new_next_layer



    # def defaultblock(self):
    #     layer= nn.Sequential(
    #         OrderedDict(
    #             [("conv1", nn.Conv2d(3, 9, 3, padding=2)),
    #              ("relu1", nn.ReLU(inplace=True)),
    #              ("max", nn.MaxPool2d(2, stride=2, ceil_mode=True))]))
    #     return layer
    def filter_selection(self, channel_num, input):
        T = []
        I = list(range(channel_num))
        while len(T) < channel_num * self.compress_rate:
            min_value = 1000
            min_id = -1
            for i in I:
                tempT = T.copy()
                tempT.append(i)
                sum = CalculSqX(input, tempT)
                if sum < min_value:
                    min_value = sum
                    min_id = i
            if min_id != -1:
                T.append(min_id)
                I.remove(min_id)
        return T

    def drop_filter(self, cur, next,cur_in_num, extra_filters):
        if(not isinstance(cur, nn.Conv2d)):
            print("The layer is not Conv2d!")
            return
        cur_out_num = cur.out_channels
        if(cur_in_num == 0):
            cur_in_num = cur.in_channels
        cur_kernel = cur.kernel_size
        cur_pad = cur.padding

        cur_new_out = cur_out_num - len(extra_filters)
        cur_new_layer = nn.Conv2d(cur_in_num, cur_new_out, cur_kernel, padding=cur_pad)

        cur_new_dict = cur_new_layer.state_dict()
        pretrained_dict = cur.state_dict()
        item = pretrained_dict.items()
        modified_dict = {}
        for k, v in item:
            new_v = v.narrow(0, 0, cur_new_out)
            j = 0
            for i in range(cur_out_num):
                if i in extra_filters:
                    continue
                new_v[j] = v[i]
                j = j + 1
            modified_dict[k] = new_v
        cur_new_dict.update(modified_dict)
        cur_new_layer.load_state_dict(cur_new_dict)

        if (not isinstance(next, nn.Conv2d)):
            print("The layer is not Conv2d!")
            return
        next_out_num = next.out_channels
        next_in_num = next.in_channels
        next_kernel = next.kernel_size
        next_pad = next.padding

        new_in = next_in_num - len(extra_filters)
        next_new_layer = nn.Conv2d(new_in, next_out_num, next_kernel, padding=next_pad)
        next_new_dict = next_new_layer.state_dict()
        next_pretrained_dict = next.state_dict()
        next_modified_dict = {}
        v = next_pretrained_dict['weight']
        new_v = v.narrow(1, 0, new_in)
        j = 0
        for i in range(next_in_num):
            if i in extra_filters:
                continue
            for l in range(next_out_num):
                new_v[l][j] = v[l][i]
            j = j + 1
        next_modified_dict['weight'] = new_v
        v = next_pretrained_dict['bias']
        next_modified_dict['bias'] = v

        next_new_dict.update(next_modified_dict)
        next_new_layer.load_state_dict(next_new_dict)

        return cur_new_layer, next_new_layer

    def save_model(self):
        torch.save(self.model,self.time_str+self.model_name)
    def get_model(self):
        return self.model


def CalculSqX(inputs, set):
    sum = 0.0
    n, c, h, w = inputs.shape
    for i in range(n):
        sub_sum = 0.0
        for j in range(c):
            if j in set:
                for k in range(h):
                    for l in range(w):
                        sub_sum +=inputs[i][j][k][l]
        sum += sub_sum*sub_sum
    return sum



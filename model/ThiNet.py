import torch.nn as nn
import torch
import time
import os
import torch.optim as optim
from torch.autograd import Variable
import utils.training as train_utils
import random
import numpy as np

class ThiNet(object):
    def __init__(self, pretrained_model, if_file = False, compress_rate = 0.1):
        self.compress_rate = compress_rate
        if(if_file):
            self.model_name = pretrained_model
            self.model = torch.load(pretrained_model)
        else:
            self.model = pretrained_model
            self.model_name = "Direct model"
        self.backed_model = self.model
        self.filter_time_rec=[]
        self.time_str = time.strftime('%y-%m-%d-%H',time.localtime(time.time()))


    def prune_layer(self, cur, next,middle_layers, cur_input, next_output):
        extra_filters = self.filter_selection(cur,next, middle_layers,cur_input, next_output)
        cur_pruned_layer,next_pruned_layer = self.drop_filter(cur, next, extra_filters)
        return cur_pruned_layer,next_pruned_layer


    def layerToPrune(self):
        modules = self.model._modules
        conv_layers=[]
        i = 0
        for name, module in modules.items():
            for layer_name, layer in module._modules.items():
                if(isinstance(layer, nn.Conv2d)):
                    conv_layers.append(i)
                i+=1
        return conv_layers


    def setTrainHelper(self, optimizer=0, criterion=0, n_epoch=1):
        local_optim = optimizer
        if(optimizer ==0):
            local_optim = optim.RMSprop(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        local_crit = criterion
        if(criterion ==0):
            local_crit = nn.MSELoss()
        self.trainHelper = train_utils.trainHelper(self.model, local_optim,local_crit, n_epochs=n_epoch)

    def thinmodel(self, train_dt, batch = 8):
        train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=False)

        print("Thinning the model...")

        modules = self.model._modules
        # Save the model's all layers
        all_layers=[]
        for name,module in modules.items():
            for layer_name, layer in module._modules.items():
                each_layer=[name,layer_name,layer]
                all_layers.append(each_layer)
        print("Total layers: ",len(all_layers))
        conv_layers=self.layerToPrune()

        # prune each_layer and save the models.
        for i in range(len(conv_layers)-1):
            self.model = self.backed_model
            pre=conv_layers[i]
            next = conv_layers[i+1]
            middle_num = next-pre-1
            middle_layers=[]
            for num in range(middle_num):
                middle_layers.append(all_layers[pre+1+num][2])

            print("Pruning the layer:", pre," The next layer: ",next)
            since = time.time()

            id_input = random.randint(0,len(train_loader)-1)
            input = 0
            for idx, data in enumerate(train_loader):
                if idx == id_input:
                    input = Variable(data[0])
                    break

            pre_input = input
            for j in range(pre):
                pre_input = all_layers[j][2](pre_input)

            next_output = pre_input
            for j in range(pre, next+1):
                next_output = all_layers[j][2](next_output)


            new_cur_layer, new_next_layer = \
                self.prune_layer(all_layers[pre][2], all_layers[next][2],
                                 middle_layers, pre_input, next_output)
            time_elapsed = time.time() - since
            print('Pruning Time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            # Save in the current model
            cur_module_name = all_layers[pre][0]
            cur_layer_name = all_layers[pre][1]
            next_module_name = all_layers[next][0]
            next_layer_name = all_layers[next][1]

            self.model._modules[cur_module_name]._modules[cur_layer_name] = new_cur_layer
            self.model._modules[next_module_name]._modules[next_layer_name] = new_next_layer

            print('Finetuning the model...')
            output = self.trainHelper.train(train_loader, if_each_acc=True, n_classes=3)
            time_elapsed = time.time() - since
            print('Total Time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            save_file = 'ThiNet/layer_'+str(pre)+'/'
            if (not os.path.exists('./'+save_file)):
                os.makedirs('./'+save_file)

            self.trainHelper.save_weights(path=save_file)
            self.trainHelper.save_results(output,train_dt,savepath=save_file)
        self.trainHelper.save_record(if_class=True,n_classes=3)



    def filter_selection(self, cur_layer, next_layer, middle_layers, cur_input, next_output):
        print("In prune_layers: filter_selection...")
        channel_num = cur_layer.out_channels
        since = time.time()
        self.filter_time_rec.append(since)

        print("The length of middle_layers is ",len(middle_layers))

        T = []
        I = list(range(channel_num))
        filter_since = since
        while len(T) < channel_num * self.compress_rate:
            print("Current to-be-dropped filter is",T)
            time_elapsed = time.time() - filter_since
            print('Filter Time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            self.filter_time_rec.append(time_elapsed)
            filter_since = time.time()

            min_value = 0
            min_id = -1
            for i in I:
                cur_output = cur_layer(cur_input)
                # set the channels in T and i as zero.
                temp_input = self.ChannelRemove(cur_output, T, i)
                for layer in middle_layers:
                    temp_input = layer(temp_input)
                temp_output = next_layer(temp_input)
                new_sum = CalculSqX(next_output, temp_output)
                if(min_value==0):
                    min_value = new_sum
                    min_id = 0
                else:
                    if new_sum < min_value:
                        min_value = new_sum
                        min_id = i
            if min_id != -1:
                T.append(min_id)
                I.remove(min_id)

        time_elapsed = time.time() - since
        print('Filter Time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        return T

    def ChannelRemove(self, arrays, set, element):
        value = arrays
        if(isinstance(arrays, Variable)):
            value = arrays.data.numpy()
        value[:,element]=0
        for ele in set:
            value[:,ele] = 0
        tensor_value = torch.from_numpy(value)
        return Variable(tensor_value)

    def drop_filter(self, cur, next, extra_filters):
        print('In prune_layer: drop_filters:',extra_filters)
        cur_out_num = cur.out_channels
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
        print("cur_out_num: ",cur_new_out)
        print("next_new_in: ",new_in)
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

    def get_model(self):
        return self.model


def CalculSqX(inputs, new_inputs):
    if(isinstance(inputs, Variable)):
        inputs = inputs.data.numpy()
        new_inputs = new_inputs.data.numpy()
    n, c, h, w = inputs.shape
    sub_sum_list = [0] * n
    sum = 0
    for i in range(n):
        for j in range(c):
            sub_sum_list[i] += np.sum(inputs[i][j]-new_inputs[i][j])
        sum += sub_sum_list[i] ** 2
    return sum, sub_sum_list
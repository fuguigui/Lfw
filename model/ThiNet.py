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
        self.time_str = time.strftime('%y-%m-%d-%H',time.localtime(time.time()))


    def prune_layer(self, new_in_num, cur, next,middle_layers, next_input):
        extra_filters = self.filter_selection(next.in_channels, next_input)
        new_out_num, cur_pruned_layer,next_pruned_layer = self.drop_filter(cur, next, new_in_num, extra_filters)
        return new_out_num, cur_pruned_layer,next_pruned_layer


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

    def thinmodel(self, train_dt, batch = 30):
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
        new_in = 0
        for i in range(len(conv_layers)-1):
            pre=conv_layers[i]
            next = conv_layers[i+1]
            middle_num = next-pre-1
            if(middle_num>0):
                middle_layers = all_layers[pre+1:pre+middle_num]
            else:
                middle_layers=[]

            print("Pruning the layer:", pre," The next layer: ",next)
            since = time.time()

            id_input = random.randint(0,len(train_loader)-1)
            input = 0
            for idx, data in enumerate(train_loader):
                if idx == id_input:
                    input = Variable(data[0])
                    break
            next_input = input

            for j in range(next):
                next_input = all_layers[j][2](next_input)

            new_in, new_cur_layer, new_next_layer = \
                self.prune_layer(new_in, all_layers[pre][2], all_layers[next][2],
                                 middle_layers,next_input)
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
            all_layers[next][2] = new_next_layer

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



    def filter_selection(self, channel_num, input):
        print("In prune_layers: filter_selection...")
        since = time.time()
        T = []
        I = list(range(channel_num))
        while len(T) < channel_num * self.compress_rate:
            print('len of T is {:d}'.format(len(T)))
            min_value = 0
            min_id = -1
            sum, sub_sum_list = CalculSqX(input, T)
            for i in I:
                new_sum = CalculSqX(input,0,sub_sum_list,new_channel=i,old_sum=sum)
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

    def drop_filter(self, cur, next,cur_in_num, extra_filters):
        print('In prune_layer: drop_filters:')
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

        return cur_new_out, cur_new_layer, next_new_layer

    def get_model(self):
        return self.model


def CalculSqX(inputs, id_list, sub_sum_list=0, new_channel = -1, old_sum=0.0):
    sum = old_sum
    if(isinstance(inputs, Variable)):
        inputs = inputs.data.numpy()
    n, c, h, w = inputs.shape
    if(new_channel!=-1):
        sub_sum=[0]*n
        for i in range(n):
            sub_sum[i] = sub_sum_list[i]
            sub_sum[i] += np.sum(inputs[i][new_channel])
            sum+=sub_sum[i]**2
        return sum
    else:
        sub_sum_list=[0]*n
        for i in range(n):
            for j in range(c):
                if j in id_list:
                    sub_sum_list[i] += np.sum(inputs[i][j])
            sum += sub_sum_list[i]**2
        return sum, sub_sum_list
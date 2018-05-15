from model import fcn
import utils.training as train_utils
from model.ThiNet import ThiNet
import torch.nn as nn
from collections import OrderedDict

fcn_model = fcn.fcn32s(n_classes=3)
new_model = ThiNet(fcn_model)

print("original model:")
print(fcn_model)
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
new_model.thinmodel()
print("Changed model:")
print(new_model.get_model())

for name, submodel in fcn_model.named_children():
    print(new_model._modules[name])


fcn_model.apply(train_utils.weights_init)

params = fcn_model.state_dict()
for k,v in params.items():
    print(k)
    bias = k.replace("weight","bias")
    print(params[k])
    print(params[bias])

print(fcn_model.parameters())
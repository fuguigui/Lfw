import model.mini_fcn as fcn
import torch.optim as optim
import utils.training as train_utils
import model.ThiNet as ThiNet
import torch.nn as nn
import utils.lfw as lfw

train_dt = lfw.Lfw("./datasets/","train_expr.txt",'/home/guigui/final_proj')


fcn_model = fcn.fcn32s()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()
train_helper = train_utils.trainHelper(fcn_model, optimizer, criterion, n_epochs=8)
train_helper.load_weights('./weights/weights-0-0.442-0.883.pth')
fcn_model = train_helper.getmodel()

fcn_thinnet = ThiNet.ThiNet(fcn_model)
fcn_thinnet.setTrainHelper()
fcn_thinnet.thinmodel(train_dt)


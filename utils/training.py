import os
import shutil
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

class trainHelper(object):
    def __init__(self, model, optimizer, criterion, learning_rate = 1e-4, LR_DECAY = 0.995, n_epochs = 10):
        self.model = model
        self.lr = learning_rate
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epoch = n_epochs
        self.RESULTS_PATH = './results'
        self.WEIGHTS_PATH = './weights'
        self.lr_decay = LR_DECAY
        self.trn_errors=[]
        self.trn_losses = []
        self.trn_eachclass_err=[]
        self.test_errors=[]
        self.test_losses=[]
        self.test_eachclass_err=[]
        self.test_best=-1
        self.time_stick = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))

    def getmodel(self):
        return self.model

    def train(self, trn_loader, if_each_acc=False, n_classes=0):
        self.model.train()
        trn_loss = 0
        trn_error = 0
        class_acc_list = [0]*n_classes

        length = len(trn_loader)
        print("In trainHelper.train: length of train loader is ", length)

        for idx,data in enumerate(trn_loader):
            inputs = Variable(data[0])
            targets = Variable(data[1])

            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, targets)

            print('Training the {:d} loader'.format(idx))

            trn_loss += loss.data[0]
            pred = self.get_predictions(output)
            right = self.get_predictions(targets)

            loss.backward()
            self.optimizer.step()

            if (if_each_acc):
                trn_err, class_acc = self.error(pred, right, n_classes)
                trn_error += trn_err
                for i in range(n_classes):
                    class_acc_list[i] += class_acc[i]
            else:
                trn_error += self.error(pred, right, 0)

        trn_loss /= len(trn_loader)
        trn_error /= len(trn_loader)
        self.trn_errors.append(trn_error)
        self.trn_losses.append(trn_loss)

        print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
            len(self.trn_errors), trn_loss, 1 - trn_error))

        if (if_each_acc):
            for i in range(n_classes):
                class_acc_list[i] /= len(trn_loader)
                print('Class:{:d}, Acc:{:.4f}'.format(i + 1, class_acc_list[i]))
            self.trn_eachclass_err.append(class_acc_list)

        return output

    def error(self, preds, targets, classes):
        assert preds.size() == targets.size()
        bs, h, w = preds.size()
        n_pixels = bs * h * w
        incorrect = preds.ne(targets).cpu().sum()
        err = incorrect / n_pixels
        if (classes == 0):
            return round(err, 5)

        preds.resize_(n_pixels)
        targets.resize_(n_pixels)

        same_list = []
        acc_class = []
        for idx in range(n_pixels):
            if (preds[idx] == targets[idx]):
                same_list.append(preds[idx])

        for i in range(classes):
            total = 0
            for j in range(n_pixels):
                if (preds[j] == i):
                    total += 1
            sub = 0
            for j in range(len(same_list)):
                if (same_list[j] == i):
                    sub += 1
            if(total==0):
                acc_class.append(0)
            else:
                acc_class.append(sub / total)
        return round(err, 5), acc_class

    def get_predictions(self, output_batch):
        bs, c, h, w = output_batch.size()
        tensor = output_batch.data
        values, indices = tensor.cpu().max(1)
        indices = indices.view(bs, h, w)
        return indices

    def test(self, test_loader, if_each_class=False, n_classes=0):
        print('In test')
        self.model.eval()
        test_loss = 0
        test_error = 0
        class_acc_list = [0]*n_classes

        for idx,data in enumerate(test_loader):
            inputs = Variable(data[0])
            target = Variable(data[1])
            print('The {:d} test loader'.format(idx))
            output = self.model(inputs)

            test_loss += self.criterion(output, target).data[0]

            pred = self.get_predictions(output)
            right = self.get_predictions(target)
            if (if_each_class):
                test_err, class_acc = self.error(pred, right, n_classes)
                test_error += test_err
                for i in range(n_classes):
                    class_acc_list[i] += class_acc[i]
            else:
                test_error += self.error(pred, right, 0)
        test_loss /= len(test_loader)
        test_error /= len(test_loader)
        self.test_losses.append(test_loss)
        self.test_errors.append(test_error)

        print('Epoch {:d}\nTest - Loss: {:.4f}, Acc: {:.4f}'.format(
            len(self.test_losses), test_loss, 1 - test_error))

        if (if_each_class):
            for i in range(n_classes):
                class_acc_list[i] /= len(test_loader)
                print('Class:{:d}, Acc:{:.4f}'.format(i + 1, class_acc_list[i]))
            self.test_eachclass_err.append(class_acc_list)


    def checkAndSave(self,output, dataset):
        if (self.test_best>-1):
            if self.test_errors[-1] < self.test_errors[self.test_best] \
                    and self.test_losses[-1] < self.test_losses[self.test_best]:
                idx = len(self.test_errors)-1
                self.test_best = idx
                self.save_weights()
                self.save_results(output, dataset)
        else:
            self.test_best = 0


    def save_weights(self,path=''):
        print("Saving weights...")

        loss = self.test_losses[self.test_best]
        err = self.test_errors[self.test_best]
        mpath = path+self.time_stick

        weights_fname = 'weights-%d-%.3f-%.3f.pth' % (self.test_best, loss, err)

        weights_fpath = os.path.join(self.WEIGHTS_PATH, mpath)
        if not os.path.exists(weights_fpath):
            os.mkdir(weights_fpath)
        weights_fpath = os.path.join(self.WEIGHTS_PATH,mpath, weights_fname)
        torch.save({
            'startEpoch': self.test_best,
            'loss': loss,
            'error': err,
            'state_dict': self.model.state_dict()
        }, weights_fpath)
        shutil.copyfile(weights_fpath, self.WEIGHTS_PATH + 'latest.th')

    def save_results(self, output,dt, savepath=''):
        print("Saving results...")

        loss = self.test_losses[self.test_best]
        err = self.test_errors[self.test_best]
        mpath = savepath+self.time_stick

        results_folder = 'results-%d-%.3f-%.3f' % (self.test_best, loss, err)
        results_fpath = os.path.join(self.RESULTS_PATH, mpath)
        if not os.path.exists(results_fpath):
            os.mkdir(results_fpath)

        results_fpath = os.path.join(self.RESULTS_PATH, mpath, results_folder)
        print('The saving path is ',results_fpath)
        for idx, item in enumerate(output):
            dt.save_output(results_fpath, idx, item)

        print("Prediction results are saved!")

    def load_weights(self, fpath):
        print("loading weights '{}'".format(fpath))
        weights = torch.load(fpath)
        startEpoch = weights['startEpoch']
        self.model.load_state_dict(weights['state_dict'])
        print("loaded weights (lastEpoch {}, loss {}, error {})"
              .format(startEpoch - 1, weights['loss'], weights['error']))

    def adjust_learning_rate(self):
        """Sets the learning rate to the initially
            configured `lr` decayed by `decay` every `n_epochs`"""
        new_lr = self.lr * (self.lr_decay ** (len(self.trn_errors) // self.n_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def FullExpr(self, train_loader, valid_loader, dt, if_classes=False, n_classes=0):
        for epoch in range(1, self.n_epoch + 1):
            since = time.time()

            print("In train_Helper - FullExpr: Traing Epoch: ",epoch)

            ### Train ###
            output = self.train(train_loader, if_each_acc=if_classes, n_classes=n_classes)
            time_elapsed = time.time() - since
            print('Train Time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            ### Valid ###
            self.test(valid_loader, if_each_class = if_classes, n_classes=n_classes)
            time_elapsed = time.time() - since
            print('Total Time {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))

            ### Checkpoint ###
            self.checkAndSave(output, dt)

            ### Adjust Lr ###
            self.adjust_learning_rate()
        print('Writing results in txt file...')
        self.save_record(if_classes, n_classes)

    def save_record(self, if_class=False, n_classes=0):
        path = './records/'
        if not os.path.exists(path):
            os.mkdir(path)
        name = os.path.join(path,self.time_stick+'_records.txt')
        f = open(name, 'w')
        f.write('# Train result:\nLosses:\n')
        f.write(str(self.trn_losses))
        f.write('\nErrors:\n')
        f.write(str(self.trn_errors))
        f.write('\nN_classes Accuracy:\n')
        f.write(str(self.trn_eachclass_err))

        f.write('\n# Test result:\nLosses:\n')
        f.write(str(self.test_losses))
        f.write('\nErrors:\n')
        f.write(str(self.test_errors))
        f.write('\nN_classes Accuracy:\n')
        f.write(str(self.test_eachclass_err))



        # for i in range(trn_len):
        #     f.write('#Rec:',i,', loss:',self.trn_losses[i],', error:',self.trn_errors[i])
        #     if(if_class):
        #         f.write('n_classes error: ')
        #         for j in range(n_classes):
        #             f.write('class ',j,': error',self.trn_eachclass_err[i][j])
        #
        # test_len = len(self.test_losses)
        # print('# Test result:\n')
        # for i in range(test_len):
        #     f.write('#Rec:', i, ', loss:', self.test_losses[i], ', error:', self.test_errors[i])
        #     if (if_class):
        #         f.write('n_classes error: ')
        #         for j in range(n_classes):
        #             f.write('class ', j, ': error', self.test_eachclass_err[i][j])
        f.close()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()
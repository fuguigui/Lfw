import os
import torch
import torch.utils.data as data
import numpy as np
import scipy.misc as m
from torchvision.datasets.folder import is_image_file
from torchvision import utils
import matplotlib.pyplot as plt
from torch.autograd import Variable

def m_loader(path):
    img = m.imread(path)
    img = np.array(img, dtype= np.uint8)
    n_img = np.pad(img,((3,3),(3,3),(0,0)),'edge')
    return n_img

def transform(img):
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def deform(result):
    if(isinstance(result, Variable)):
        result = result.data
    result = result.numpy()
    result = result.transpose(1,2,0)
    result = result[3:253, 3:253]
    idx = np.argmax(result, axis=2)
    h,w = idx.shape
    res_2 = np.zeros((h,w,3))
    for i in range(h):
        for j in range(w):
            k = idx[i][j]
            res_2[i][j][k] = 255
    res = res_2.astype(int)
    return res

def _make_dataset(root,txt,dir):
    images = []
    labels = []
    f = open(root+txt)
    for line in f.readlines():
        img, label = line.rstrip().split('\t')
        img = img.replace('..',dir)
        label = label.replace('..',dir)
        if is_image_file(img):
            images.append(img)
        if is_image_file(label):
            labels.append(label)
    return images, labels

class Lfw(data.Dataset):

    def __init__(self, root,txt, dir,transform = transform,
                 loader=m_loader, n_classes = 3):
        self.type = type
        self.n_classes = n_classes
        self.loader = loader
        self.imgs, self.labs = _make_dataset(root,txt,dir)
        self.height = 250
        self.weight = 250
        self.transform = transform

    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        lab = self.loader(self.labs[index])
        img = self.transform(img)
        lab = self.transform(lab)
        return img, lab

    def __len__(self):
        return len(self.imgs)

    def save_output(self,path, idx, img):
        out = deform(img)
        original_file = self.imgs[idx]
        fname = original_file.split('/')[-1]
        fname = fname.replace('.jpg','_pred.ppm')
        print('In lfw: saving name is ',fname)

        if not os.path.exists(path):
            os.mkdir(path)
        m.imsave(os.path.join(path, fname), out)



def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    plt.title("Batch from dataloader")


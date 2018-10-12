import torch
import torch.utils.data as data
import numpy as np
import scipy.misc as m
from torchvision.datasets.folder import is_image_file
from torchvision import utils
import matplotlib.pyplot as plt

def m_loader(path):
    img = m.imread(path)
    return np.array(img,dtype=np.uint8)

def transform(img, lbl):
    img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()

    lbl = lbl.astype(float) / 255.0
    lbl = lbl.transpose(2, 0, 1)
    lbl = torch.from_numpy(lbl).float()
    return img, lbl


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
        img, lab = self.transform(img, lab)
        return img, lab

    def __len__(self):
        return len(self.imgs)


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    plt.title("Batch from dataloader")


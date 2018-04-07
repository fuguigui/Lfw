import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import scipy.misc as m
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import utils
import matplotlib.pyplot as plt
import torchvision.transforms as transform


classes = ['skin','background','hair']

class_weight = torch.FloatTensor([
    0.58872014284134, 0.51052379608154, 2.6966278553009,
    0.45021694898605, 1.1785038709641, 0.77028578519821, 2.4782588481903,
    2.5273461341858, 1.0122526884079, 3.2375309467316, 4.1312313079834, 0])

mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

class_color = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]

def my_loader(path):
    return Image.open(path).convert('RGB')

def m_loader(path):
    img = m.imread(path)
    return np.array(img,dtype=np.uint8)

def _make_dataset(root,txt):
    images = []
    labels = []
    f = open(root+txt)
    for line in f.readlines():
        img, label = line.rstrip().split('\t')
        img = img.replace('..','/home/guigui/final_proj')
        label = label.replace('..','/home/guigui/final_proj')
        if is_image_file(img):
            images.append(img)
        if is_image_file(label):
            labels.append(label)
    return images, labels


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


class LabelTensorToPILImage(object):
    def __call__(self, label):
        label = label.unsqueeze(0)
        colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        for i, color in enumerate(class_color):
            mask = label.eq(i)
            for j in range(3):
                colored_label[j].masked_fill_(mask, color[j])
        npimg = colored_label.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"

        return Image.fromarray(npimg, mode=mode)


class Lfw(data.Dataset):

    def __init__(self, dir,txt, transform=None, target_transform=LabelToLongTensor,
                 loader=m_loader):
        self.type = type
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs, self.labs = _make_dataset(dir,txt)

    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        lab = self.loader(self.labs[index])
        print("origin img = ",img,
              "\norigin lab =",lab)

        plt.imshow(img)
        plt.show()
        plt.imshow(lab)
        plt.show()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            lab = self.target_transform(lab)
        print("Transforms:\nimg = ",img,
              "\n lab = ",lab)
        return img, lab

    def __len__(self):
        return len(self.imgs)


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    plt.title("Batch from dataloader")

import os
import numpy as np
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import scipy.misc as m
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import utils
import matplotlib.pyplot as plt

Bkg = [0,0,255]
Hair = [255, 0, 0]
Skin = [0, 255, 0]
Unlbl = [0, 0, 0]
classes = np.array([Bkg, Hair, Skin, Unlbl])

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
        for i, color in enumerate(classes):
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

    def __init__(self, dir,txt, is_transform = True,
                 if_encode=False,
                 loader=m_loader, n_classes = 3):
        self.type = type
        self.n_classes = n_classes
        self.is_transform = is_transform
        self.loader = loader
        self.imgs, self.labs = _make_dataset(dir,txt)
        self.height = 250
        self.weight = 250
        self.if_encode = if_encode

    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        lab = self.loader(self.labs[index])
        #print("origin img = ",img,
         #     "\norigin lab =",lab)

        #plt.imshow(img)
        #plt.show()
        #plt.imshow(lab)
        #plt.show()

        if self.if_encode:
            lab = encode_segmap(lab, self.height, self.weight)

        if self.is_transform:
            img, lab = self.transform(img, lab)

        #print("Transforms:\nimg = ",img,
         #     "\n lab = ",lab)
        return img, lab

    def __len__(self):
        return len(self.imgs)

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        # img -= self.mean
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

def encode_segmap(lab, x, y):
    new_lab = np.zeros((x, y,1))
    for h in range(x):
        for w in range(y):
            for i, color in enumerate(classes):
                if lab[h][w][0] == color[0] and lab[h][w][1] == color[1] and lab[h][w][2] == color[2]:
                    new_lab[h][w][0] = i
                    break
    return new_lab
def decode_segmap(lab):
    return lab

def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    plt.title("Batch from dataloader")


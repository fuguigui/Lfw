import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# FCN32s
class fcn32s(nn.Module):
    def __init__(self, n_classes=3, learned_bilinear=False):
        super(fcn32s,self).__init__()
        self.learned_bilinear = learned_bilinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            OrderedDict([
                ("conv1",nn.Conv2d(3,64,3,padding=4)),
                ("relu1",nn.ReLU(inplace=True)),
                ("conv2",nn.Conv2d(64,64,3,padding=1)),
                ("relu2",nn.ReLU(inplace=True)),
                ("max",nn.MaxPool2d(2,stride=2,ceil_mode=True))
            ]))
        self.conv_block2 = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(64, 128, 3, padding=1)),
                ("relu1", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(128, 128, 3, padding=1)),
                ("relu2", nn.ReLU(inplace=True)),
                ("max", nn.MaxPool2d(2, stride=2, ceil_mode=True))
            ]))

        self.conv_block3 = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(128, 256, 3, padding=1)),
                ("relu1", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(256, 256, 3, padding=1)),
                ("relu2", nn.ReLU(inplace=True)),
                ("conv3", nn.Conv2d(256, 256, 3, padding=1)),
                ("relu3", nn.ReLU(inplace=True)),
                ("max", nn.MaxPool2d(2, stride=2, ceil_mode=True))
            ]))
        self.conv_block4 = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(256, 512, 3, padding=1)),
                ("relu1", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(512, 512, 3, padding=1)),
                ("relu2", nn.ReLU(inplace=True)),
                ("conv3", nn.Conv2d(512, 512, 3, padding=1)),
                ("relu3", nn.ReLU(inplace=True)),
                ("max", nn.MaxPool2d(2, stride=2, ceil_mode=True))
            ]))
        self.classifier = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(512,1024,4, padding=1)),
                ("relu1", nn.ReLU(inplace=True)),
                ("drop1", nn.Dropout2d(inplace=True)),
                ("conv2", nn.Conv2d(1024,1024, 1, padding=0)),
                ("relu2", nn.ReLU(inplace=True)),
                ("drop2", nn.Dropout2d(inplace=True)),
                ("conv3", nn.Conv2d(1024,self.n_classes,1))]))

    def forward(self,x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        score = self.classifier(conv4)
        out = F.upsample(score, x.size()[2:])

        return out

# FCN16s
class fcn16s(nn.Module):
    def __init__(self, n_classes=3, learned_bilinear=False):
        super(fcn16s, self).__init__()
        self.learned_bilinear = learned_bilinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, self.n_classes, 1),
        )


        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_bilinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        score = self.classifier(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3
        out = F.upsample(score, x.size()[2:])

        return out


# FCN8s
class fcn8s(nn.Module):
    def __init__(self, n_classes=3, learned_bilinear=False):
        super(fcn8s, self).__init__()
        self.learned_bilinear = learned_bilinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, self.n_classes, 1),
        )

        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        self.score_pool2 = nn.Conv2d(128, self.n_classes, 1)

        if self.learned_bilinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        score = self.classifier(conv4)
        score_pool3 = self.score_pool3(conv3)
        score_pool2 = self.score_pool2(conv2)

        score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3
        score = F.upsample(score, score_pool2.size()[2:])
        score += score_pool2
        out = F.upsample(score, x.size()[2:])

        return out

import torch
from torch import nn
import torch.nn.functional as F
# import visdom

class ST(nn.Module):
    def __init__(self, inplanes, planes, inshape, outshape):
        super(ST, self).__init__()
        self.inshape = inshape
        self.outshape = outshape

        self.vs = visdom.Visdom(port=7666)

        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.relu3= nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.relu4= nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(8 , stride=1)
       
        self.fc0 = nn.Linear(planes, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(16, 6)


    def localization(self, x):

        xs = self.maxpool1(x)
        xs = self.conv1(xs)
        xs = self.relu1(xs)

        xs = self.conv2(xs)
        xs = self.relu2(xs)

        xs = self.maxpool2(xs)

        xs = self.conv3(xs)
        xs = self.relu3(xs)

        xs = self.conv4(xs)
        xs = self.relu4(xs)

        xs = self.avgpool(xs)
        xs = xs.view(xs.size(0), -1)
        xs = self.fc0(xs)
        xs = self.relu3(xs)
        xs = self.fc1(xs)
        xs = self.relu4(xs)
        xs = self.fc(xs)

        return xs

    def st(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size([x.size(0), x.size(1), self.outshape, self.outshape]))
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        xs = self.st(x)
        out = x + xs
        out = (out - torch.mean(out))/torch.std(out)
        # self.vs.images(x.data.cpu().numpy(), win=1, opts={"title": "The original input images"})
        # self.vs.images(xs.data.cpu().numpy(), win=2, opts={"title": "The warped input images"})
        # self.vs.images(out.data.cpu().numpy(), win=3, opts={"title": "Merged image"})

        return out

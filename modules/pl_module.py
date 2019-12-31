from torch import nn
import torch
import torch.nn.functional as F
import visdom
import numpy as np

class PLLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PLLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(channel * 2, channel, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                # nn.Softmax(1)
        )

    def forward(self, xin, xout):
        b, c, _, _ = xin.size()
        # aggregation
        x = torch.cat([xin, xout], 1)
        # compute
        x = self.conv1x1(x)
        x = self.relu(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return xout * y

class PLminus(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PLminus, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(channel, affine=False)
        self.relu = nn.ReLU()
        self.pdist = nn.PairwiseDistance(p=2,keepdim=True)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                # nn.Softmax(1)
                # nn.Tanh()
        )


    def forward(self, xin, xout):
        b, c, _, _ = xin.size()
        # aggregation
        # x = torch.cat([xin, xout], 1)
        # x = xin + xout

        # similarity:
        # minus
        # x = self.relu(self.bn(xout)) - self.relu(self.bn(xin))

        # L2-avg
        x1 = self.relu(xin)
        x1 = self.avg_pool(x1)
        x2 = self.relu(xout)
        x2 = self.avg_pool(x2)
        x = (x1 - x2)**2
        y = x.view(b, c)
        # compute
        # y = self.avg_pool(x).view(b, c)
        # todo: add bias
        y = self.fc(y).view(b, c, 1, 1)
        return self.relu(self.bn(xout) * y * 2)

class PLasConv(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PLasConv, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, groups=channel)
        self.bn = nn.BatchNorm2d(channel, affine=False)
        self.relu = nn.ReLU()

        self.conv1x1.weight.data.fill_(1)
        self.conv1x1.bias.data.zero_()

    def forward(self, xin, xout=None):

        if (xout is not None):
            x = xin + xout
        else:
            x = xin

        # method (1):
        # x = self.bn(x)
        # method (2):
        x = (x - x.mean(0, True)) / (x.std(0, True) + 1e-5)
        # method (3):
        # norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+ 1e-6
        # x = torch.div(x,norm)

        x = self.conv1x1(x)
        x = self.relu(x)

        return x


class ResPL(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ResPL, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, groups=channel)
        # self.bn = nn.BatchNorm2d(channel, affine=False)
        self.relu = nn.ReLU()

        self.conv1x1.weight.data.fill_(1)
        self.conv1x1.bias.data.zero_()

    def forward(self, xin, xout=None):

        if (xout is not None):
            x = xin + xout
        else:
            x = xin

        xin = xin / torch.norm(xin)
        xout = xout / torch.norm(xout)

        # method (1):
        # x = self.bn(x)
        # method (2):
        # x = (x - x.mean(0, True)) / (x.std(0, True) + 1e-5)
        # method (3):
        # norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+ 1e-6
        # x = torch.div(x,norm)

        xout = self.conv1x1(xout)
        x = torch.cat((xin, xout), dim=1)
        # x = self.relu(x)

        return x

class STPL(nn.Module):
    def __init__(self, shape, channel, reduction=16):
        super(STPL, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, groups=channel)
        # self.bn = nn.BatchNorm2d(channel, affine=False)
        self.relu = nn.ReLU()
        self.vis = visdom.Visdom(port=7777)
        # self.stavg = nn.AdaptiveAvgPool2d(1)
        self.stfc1 = nn.Linear(channel * shape * shape, 32)
        self.stfc2 = nn.Linear(32, 32)
        self.stfc3 = nn.Linear(32, 6)
        self.pl_lambda = nn.Linear(32, channel)
        self.pl_s = nn.Linear(6, 1)


        self.stfc3.weight.data.zero_()
        self.stfc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.conv1x1.weight.data.fill_(1)
        self.conv1x1.bias.data.zero_()
        self.pl_s.weight.data.zero_()
        self.pl_s.bias.data.fill_(1)

    def forward(self, x):


        xs = x.view(x.size(0), -1)
        xs = self.relu(self.stfc1(xs))
        xs = self.relu(self.stfc2(xs))
        theta = self.stfc3(xs)

        s   = self.pl_s(theta).mean()
        lam = self.pl_lambda(xs).mean(0, True)

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        x = x * torch.exp(s * lam * -1).view(1, -1, 1, 1)

        self.vis.line(Y=s.data.cpu().numpy().reshape(1), X=np.array([1]), win="PL s", update="append")
        self.vis.line(Y=lam.median().data.cpu().numpy().reshape(1), X=np.array([1]), win="PL lam", update="append")
        self.vis.line(Y=torch.exp(s * lam * -1).median().data.cpu().numpy().reshape(1), X=np.array([1]), win="factor", update="append")


        x = self.conv1x1(x)
        # x = self.relu(x)

        return x
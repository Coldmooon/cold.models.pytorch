from torch import nn
import torch

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
        y = self.fc(y).view(b, c, 1, 1)
        return self.relu(self.bn(xout * y * 2))

class PLasConv(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PLasConv, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, groups=channel)
        self.bn = nn.BatchNorm2d(channel, affine=False)
        self.relu = nn.ReLU()

        self.conv1x1.weight.data.fill_(1)
        self.conv1x1.bias.data.zero_()

    def forward(self, xin, xout=None):
        # b, c, _, _ = xin.size()

        if (xout is not None):
            x = xin + xout
        else:
            x = xin

        x = self.bn(x)
        x = self.conv1x1(x)
        x = self.relu(x)

        return x
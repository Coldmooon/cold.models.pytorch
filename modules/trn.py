import torch
from torch import nn
import torch.nn.functional as F
import visdom
from augmenfly import AUGFLY

class PL(nn.Module):
    def __init__(self, batch, channel, reduction=1):
        super(PL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                # nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm1d(channel // reduction),
                # nn.LayerNorm(torch.Size([channel // reduction, 1, 1])),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                # nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm1d(channel),
                # nn.LayerNorm(torch.Size([channel, 1, 1])),
                nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.Tensor(1, channel, 1, 1), requires_grad=True).cuda().data.zero_()

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y, self.bias

class TRN(nn.Module):
    def __init__(self, inshape, outshape, inchannels, rot):
        super(TRN, self).__init__()
        self.pl = PL(batch=64, channel=inchannels)
        self.norm = nn.BatchNorm2d(num_features=inchannels, affine=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        normed = self.norm(x)
        r, b = self.pl(self.relu(normed))

        return normed * r + b
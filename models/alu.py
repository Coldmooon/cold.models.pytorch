# https://github.com/moskomule/senet.pytorch/blob/master/se_module.py
from torch import nn
# import visdom
import numpy as np
import torch

class ALU(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALU, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                # nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(channel // reduction),
                # nn.LayerNorm(torch.Size([channel // reduction, 1, 1])),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                # nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(channel),
                # nn.LayerNorm(torch.Size([channel, 1, 1])),
                nn.Sigmoid()
                # nn.Tanh()
        )
        # self.count = 0
        # self.vis = visdom.Visdom(port=7777)

    def forward(self, x):
        b, c, _, _ = x.size()
        maskneg = (x < 0).float()
        maskpos = (x > 0).float()

        y = self.avg_pool(self.relu(x)).view(b, c)
        # y = self.avg_pool(self.relu(x))
        y = self.fc(y).view(b, c, 1, 1)
        # z = maskneg * y + maskpos

        # if (self.count % 2000 == 0):
        #     if (self.count == 0):
        #         self.vis.line(X=np.array([self.count]), Y=y[0].mean().data.cpu().numpy().reshape(1), win="line2")
        #     else:
        #         self.vis.line(X=np.array([self.count]), Y=y[0].mean().data.cpu().numpy().reshape(1), win="line2", update="append")

        # self.count += 1
        return x * (maskneg * y + maskpos)

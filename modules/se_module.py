# https://github.com/moskomule/senet.pytorch/blob/master/se_module.py
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                # nn.BatchNorm1d(channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                # nn.BatchNorm1d(channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

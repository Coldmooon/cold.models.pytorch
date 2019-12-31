import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch

from trn import TRN
from augmenfly import AUGFLY
from st_module import ST

__all__ = ['TransError', 'plinst']

class TransError(nn.Module):
    def __init__(self, r =16):
        super(TransError, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.stn = ST(inplanes=3, planes=20, inshape=26, outshape=26)
        # self.dataaug = AUGFLY(28, 28, rot=(-3.1415926, 3.1415926))
        # self.conv1recf = TRN(26, 26, 3, rot=(-3.1415926, 3.1415926))
        self.conv1aug = AUGFLY(26, 26, rot=(-3.1415926, 3.1415926))
        # self.conv2recf = TRN(24, 24, 6, rot=(-3.1415926, 3.1415926))
        # self.conv2aug = AUGFLY(12, 12, rot=(-3.1415926, 3.1415926))
        # self.conv3recf = TRN(10, 10, 9, rot=(-3.1415926, 3.1415926))
        # self.conv3aug = AUGFLY(10, 10, rot=(-3.1415926, 3.1415926))
        # self.conv4recf = TRN(8, 8, 12, rot=(-3.1415926, 3.1415926))
        # self.conv5recf = TRN(8, 8, 15, rot=(-3.1415926, 3.1415926))
        # self.conv6recf = TRN(8, 8, 18, rot=(-3.1415926, 3.1415926))

        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=0),
            # nn.BatchNorm2d(num_features=3),
            # self.conv1recf,
            nn.ReLU(),
            # self.conv1aug
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=0),
            # nn.BatchNorm2d(num_features=6),
            # self.conv2recf,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # nn.Dropout(0.5)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, padding=0),
            # nn.BatchNorm2d(num_features=9),
            # self.conv3recf,
            nn.ReLU(),
            # self.conv3aug,
            # self.conv3aug
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3, padding=0),
            # nn.BatchNorm2d(num_features=12),
            # self.conv4recf,
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=15, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=192)
            # self.conv5recf,
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        # hidden6 = nn.Sequential(
        #     nn.Conv2d(in_channels=15, out_channels=18, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(num_features=192),
        #     # self.conv6recf,
        #     nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        #     # nn.Dropout(0.5)
        # )
        # hidden7 = nn.Sequential(
        #     nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(num_features=192),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     # nn.Dropout(0.5)
        # )
        # hidden8 = nn.Sequential(
        #     nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(num_features=192),
        #     nn.ReLU(),
        #     self.pl8,
        #     # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        #     nn.Dropout(0.5)
        # )
        hidden9 = nn.Sequential(
            # nn.Linear(192 * 7 * 7, 3072),
            nn.Linear(15 * 8 * 8, 48),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        # hidden10 = nn.Sequential(
        #     nn.Linear(3072, 3072),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )

        hidden11 = nn.Sequential(
            nn.Linear(48, 10),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self._features = nn.Sequential(
            hidden1,
            self.conv1aug,
            self.stn,
            hidden2,

            hidden3,

            hidden4,
            hidden5,
            # hidden6
        )

        self._classifier = nn.Sequential(
            hidden9,
            # hidden10,
            hidden11
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        # x = self.dataaug(x)
        # x = self.stn(x)
        x = self._features(x)
        x = x.view(x.size(0), -1)
        score = self._classifier(x)

        return score

def transerror(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TransError()
    return model

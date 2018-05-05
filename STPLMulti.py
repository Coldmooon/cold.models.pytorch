import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
from se_module import SELayer

__all__ = ['STPLMulti', 'stm11', 'baseline']


class STPLMulti(nn.Module):
    def __init__(self, r =16):
        super(STPLMulti, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.se1 = SELayer(48, r)
        self.se2 = SELayer(64, r)
        self.se3 = SELayer(128, r)
        self.se4 = SELayer(160, r)
        self.se5 = SELayer(192, r)
        self.se6 = SELayer(192, r)
        self.se7 = SELayer(192, r)
        self.se8 = SELayer(192, r)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            self.se4,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se5,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se6,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se7,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se8,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            # nn.Linear(192 * 7 * 7, 3072),
            nn.Linear(192 * 3 * 3, 3072),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        hidden11 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9,
            hidden10,
            hidden11
        )

        self._digit_length = nn.Sequential(nn.Linear(3072, 7))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        self.st1fc1 = nn.Linear(54 * 54 * 3, 32)
        self.st1fc2 = nn.Linear(32, 32)
        self.st1fc3 = nn.Linear(32, 6)

        self.st2fc1 = nn.Linear(27 * 27 * 48, 32)
        self.st2fc2 = nn.Linear(32, 32)
        self.st2fc3 = nn.Linear(32, 6)

        self.st3fc1 = nn.Linear(27 * 27 * 64, 32)
        self.st3fc2 = nn.Linear(32, 32)
        self.st3fc3 = nn.Linear(32, 6)

        self.st4fc1 = nn.Linear(13 * 13 * 128, 32)
        self.st4fc2 = nn.Linear(32, 32)
        self.st4fc3 = nn.Linear(32, 6)

        self.st1fc3.weight.data.zero_()
        self.st1fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st2fc3.weight.data.zero_()
        self.st2fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st3fc3.weight.data.zero_()
        self.st3fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st4fc3.weight.data.zero_()
        self.st4fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def st1(self, x):

        xs = x.view(x.size(0), -1)
        xs = self.relu(self.st1fc1(xs))
        xs = self.relu(self.st1fc2(xs))
        theta = self.st1fc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def st2(self, x):

        xs = x.view(x.size(0), -1)
        xs = self.relu(self.st2fc1(xs))
        xs = self.relu(self.st2fc2(xs))
        theta = self.st2fc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def st3(self, x):

        xs = x.view(x.size(0), -1)
        xs = self.relu(self.st3fc1(xs))
        xs = self.relu(self.st3fc2(xs))
        theta = self.st3fc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def st4(self, x):

        xs = x.view(x.size(0), -1)
        xs = self.relu(self.st4fc1(xs))
        xs = self.relu(self.st4fc2(xs))
        theta = self.st4fc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.st1(x)
        x = self.hidden1(x)
        x = self.st2(x)
        x = self.se1(x)
        x = self.hidden2(x)
        x = self.st3(x)
        x = self.se2(x)
        x = self.hidden3(x)
        x = self.st4(x)
        x = self.se3(x)
        x = self.hidden4(x)
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self._classifier(x)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits


def stpl11(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STPLMulti(r=r)
    return model


import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
from pl_module import PLLayer
from pl_module import PLminus
from pl_module import PLasConv
from se_module import SELayer
from pl_module import ResPL
from pl_module import STPL


__all__ = ['STPLMulti', 'STPLFCNMulti', 'STPLFCNv3', 'STPLFCNv4','STPLminus', 'PLinConv', 'ResPLNet', 'PLinST',
           'stpl11', 'stplfcn9', 'stplagg', 'stplgroupconv', 'stplminus', 'stplinconv', 'respl', 'plinst']


class STPLMulti(nn.Module):
    def __init__(self, r =16):
        super(STPLMulti, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pl1 = PLLayer(48, r)
        self.pl2 = PLLayer(64, r)
        self.pl3 = PLLayer(128, r)
        self.pl4 = PLLayer(160, r)
        self.pl5 = PLLayer(192, r)
        self.pl6 = PLLayer(192, r)
        self.pl7 = PLLayer(192, r)
        self.pl8 = PLLayer(192, r)

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
            self.pl4,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl5,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl6,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl7,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl8,
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
        x = self.pl1(x)
        x = self.hidden2(x)
        x = self.st3(x)
        x = self.pl2(x)
        x = self.hidden3(x)
        x = self.st4(x)
        x = self.pl3(x)
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


class STPLFCNMulti(nn.Module):
    def __init__(self, r =16):
        super(STPLFCNMulti, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pl1 = PLLayer(48, r)
        self.pl2 = PLLayer(64, r)
        self.pl3 = PLLayer(128, r)
        self.pl4 = PLLayer(160, r)
        self.pl5 = PLLayer(192, r)
        self.pl6 = PLLayer(192, r)
        self.pl7 = PLLayer(192, r)
        self.pl8 = PLLayer(192, r)

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
            self.pl4,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl5,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl6,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl7,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.pl8,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.pl1(x)
        x = self.hidden2(x)
        x = self.st3(x)
        x = self.pl2(x)
        x = self.hidden3(x)
        x = self.st4(x)
        x = self.pl3(x)
        x = self.hidden4(x)
        x = self._features(x)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits

class STPLFCNv3(nn.Module):
    def __init__(self, r =16):
        super(STPLFCNv3, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pl1 = PLLayer(48, r)
        self.pl2 = PLLayer(64, r)
        self.pl3 = PLLayer(128, r)
        self.se4 = SELayer(160, r)
        self.se5 = SELayer(192, r)
        self.se6 = SELayer(192, r)
        self.se7 = SELayer(192, r)
        self.se8 = SELayer(192, r)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            self.se4,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se5,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se6,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se7,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se8,
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        xout1 = self.st2(x)
        x_rec1 = self.pl1(x, xout1)
        x_rec1 = self.hidden2(x_rec1)
        xout2 = self.st3(x_rec1)
        x_rec2 = self.pl2(x_rec1, xout2)
        x_rec2 = self.hidden3(x_rec2)
        xout3 = self.st4(x_rec2)
        x_rec3 = self.pl3(x_rec2, xout3)
        x_rec3 = self.hidden4(x_rec3)
        x = self._features(x_rec3)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits


class STPLFCNv4(nn.Module):
    def __init__(self, r =16):
        super(STPLFCNv4, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        self.pl1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, groups=48, bias=False),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        self.pl2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, groups=64, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        self.pl3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, groups=128, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.st1fc1 = nn.Linear(54 * 54 * 3, 32)
        # self.st1fc2 = nn.Linear(32, 32)
        # self.st1fc3 = nn.Linear(32, 6)

        self.st2fc1 = nn.Linear(27 * 27 * 48, 32)
        self.st2fc2 = nn.Linear(32, 32)
        self.st2fc3 = nn.Linear(32, 6)

        self.st3fc1 = nn.Linear(27 * 27 * 64, 32)
        self.st3fc2 = nn.Linear(32, 32)
        self.st3fc3 = nn.Linear(32, 6)

        self.st4fc1 = nn.Linear(13 * 13 * 128, 32)
        self.st4fc2 = nn.Linear(32, 32)
        self.st4fc3 = nn.Linear(32, 6)

        # self.st1fc3.weight.data.zero_()
        # self.st1fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st2fc3.weight.data.zero_()
        self.st2fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st3fc3.weight.data.zero_()
        self.st3fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st4fc3.weight.data.zero_()
        self.st4fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # def st1(self, x):
    #
    #     xs = x.view(x.size(0), -1)
    #     xs = self.relu(self.st1fc1(xs))
    #     xs = self.relu(self.st1fc2(xs))
    #     theta = self.st1fc3(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #
    #     return x

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
        # x = self.st1(x)
        x = self.hidden1(x)
        x = self.st2(x)
        x = self.pl1(x)
        x = self.hidden2(x)
        x = self.st3(x)
        x = self.pl2(x)
        x = self.hidden3(x)
        x = self.st4(x)
        x = self.pl3(x)

        x = self.hidden4(x)
        x = self._features(x)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits

class STPLminus(nn.Module):
    def __init__(self, r =16):
        super(STPLminus, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.pl1 = PLminus(48, r)
        self.pl2 = PLminus(64, r)
        self.pl3 = PLminus(128, r)
        self.se4 = SELayer(160, r)
        self.se5 = SELayer(192, r)
        self.se6 = SELayer(192, r)
        self.se7 = SELayer(192, r)
        self.se8 = SELayer(192, r)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=48),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        # self.pl1 = nn.Sequential(
        #     # nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, groups=48, bias=False),
        #     PLLayer(48, r)
        # )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        # self.pl2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, groups=64, bias=False),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU()
        # )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=128),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        # self.pl3 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, groups=128, bias=False),
        #     nn.BatchNorm2d(num_features=128),
        #     nn.ReLU()
        # )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            self.se4
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se5,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se6
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se7,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            self.se8
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         # m.bias.data.zero_()

        # self.st1fc1 = nn.Linear(54 * 54 * 3, 32)
        # self.st1fc2 = nn.Linear(32, 32)
        # self.st1fc3 = nn.Linear(32, 6)

        self.st2fc1 = nn.Linear(27 * 27 * 48, 32)
        self.st2fc2 = nn.Linear(32, 32)
        self.st2fc3 = nn.Linear(32, 6)

        self.st3fc1 = nn.Linear(27 * 27 * 64, 32)
        self.st3fc2 = nn.Linear(32, 32)
        self.st3fc3 = nn.Linear(32, 6)

        self.st4fc1 = nn.Linear(13 * 13 * 128, 32)
        self.st4fc2 = nn.Linear(32, 32)
        self.st4fc3 = nn.Linear(32, 6)

        # self.st1fc3.weight.data.zero_()
        # self.st1fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st2fc3.weight.data.zero_()
        self.st2fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st3fc3.weight.data.zero_()
        self.st3fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st4fc3.weight.data.zero_()
        self.st4fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # def st1(self, x):
    #
    #     xs = x.view(x.size(0), -1)
    #     xs = self.relu(self.st1fc1(xs))
    #     xs = self.relu(self.st1fc2(xs))
    #     theta = self.st1fc3(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #
    #     return x

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
        # x = self.st1(x)
        xin1 = self.hidden1(x)
        xout1 = self.st2(xin1)
        rec1 = self.pl1(xin1, xout1)

        # rec1 = self.maxpool(rec1)

        xin2 = self.hidden2(rec1)
        xout2 = self.st3(xin2)
        rec2 = self.pl2(xin2, xout2)

        xin3 = self.hidden3(rec2)
        xout3 = self.st4(xin3)
        rec3 = self.pl3(xin3, xout3)

        # rec3 = self.maxpool(rec3)

        xin4 = self.hidden4(rec3)
        x = self._features(xin4)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits


class PLinConv(nn.Module):
    def __init__(self, r =16):
        super(PLinConv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn= nn.BatchNorm2d(num_features=48)

        self.pl1 = PLasConv(48, r)
        self.pl2 = PLasConv(64, r)
        self.pl3 = PLasConv(128, r)
        self.pl4 = PLasConv(160, r)
        self.pl5 = PLasConv(192, r)
        self.pl6 = PLasConv(192, r)
        self.pl7 = PLasConv(192, r)
        self.pl8 = PLasConv(192, r)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=48),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=160),
            self.pl4,
            # nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            self.pl5,
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            self.pl6,
            # nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            self.pl7,
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            self.pl8,
            # nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         # m.bias.data.zero_()

        # self.st1fc1 = nn.Linear(54 * 54 * 3, 32)
        # self.st1fc2 = nn.Linear(32, 32)
        # self.st1fc3 = nn.Linear(32, 6)

        self.st2avg = nn.AdaptiveAvgPool2d(1)
        self.st2fc1 = nn.Linear(48, 32)
        self.st2fc2 = nn.Linear(32, 32)
        self.st2fc3 = nn.Linear(32, 6)

        self.st3avg = nn.AdaptiveAvgPool2d(1)
        self.st3fc1 = nn.Linear(64, 32)
        self.st3fc2 = nn.Linear(32, 32)
        self.st3fc3 = nn.Linear(32, 6)

        self.st4avg = nn.AdaptiveAvgPool2d(1)
        self.st4fc1 = nn.Linear(128, 32)
        self.st4fc2 = nn.Linear(32, 32)
        self.st4fc3 = nn.Linear(32, 6)

        # self.st1fc3.weight.data.zero_()
        # self.st1fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st2fc3.weight.data.zero_()
        self.st2fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st3fc3.weight.data.zero_()
        self.st3fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st4fc3.weight.data.zero_()
        self.st4fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # def st1(self, x):
    #
    #     xs = x.view(x.size(0), -1)
    #     xs = self.relu(self.st1fc1(xs))
    #     xs = self.relu(self.st1fc2(xs))
    #     theta = self.st1fc3(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #
    #     return x

    def st2(self, x):

        xs = self.st2avg(x)
        xs = xs.view(xs.size(0), -1)
        xs = self.relu(self.st2fc1(xs))
        xs = self.relu(self.st2fc2(xs))
        theta = self.st2fc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def st3(self, x):
        xs = self.st3avg(x)
        xs = xs.view(xs.size(0), -1)
        xs = self.relu(self.st3fc1(xs))
        xs = self.relu(self.st3fc2(xs))
        theta = self.st3fc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def st4(self, x):
        xs = self.st4avg(x)
        xs = xs.view(xs.size(0), -1)
        xs = self.relu(self.st4fc1(xs))
        xs = self.relu(self.st4fc2(xs))
        theta = self.st4fc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # x = self.st1(x)
        xin1 = self.hidden1(x)
        xout1 = self.st2(xin1)
        rec1 = self.pl1(xin1, xout1)

        rec1 = self.maxpool(rec1)

        xin2 = self.hidden2(rec1)
        xout2 = self.st3(xin2)
        rec2 = self.pl2(xin2, xout2)

        xin3 = self.hidden3(rec2)
        xout3 = self.st4(xin3)
        rec3 = self.pl3(xin3, xout3)

        rec3 = self.maxpool(rec3)

        xin4 = self.hidden4(rec3)
        x = self._features(xin4)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits

class ResPLNet(nn.Module):
    def __init__(self, r =16):
        super(ResPLNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn= nn.BatchNorm2d(num_features=48)

        self.pl1 = ResPL(24, r)
        self.pl2 = ResPL(32, r)
        self.pl3 = ResPL(64, r)
        # self.pl4 = ResPL(160, r)
        # self.pl5 = ResPL(192, r)
        # self.pl6 = ResPL(192, r)
        # self.pl7 = ResPL(192, r)
        # self.pl8 = ResPL(192, r)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=160),
            # self.pl4,
            # nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl5,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl6,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl7,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl8,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         # m.bias.data.zero_()

        # self.st1fc1 = nn.Linear(54 * 54 * 3, 32)
        # self.st1fc2 = nn.Linear(32, 32)
        # self.st1fc3 = nn.Linear(32, 6)

        self.st2fc1 = nn.Linear(27 * 27 * 24, 32)
        self.st2fc2 = nn.Linear(32, 32)
        self.st2fc3 = nn.Linear(32, 6)

        self.st3fc1 = nn.Linear(27 * 27 * 32, 32)
        self.st3fc2 = nn.Linear(32, 32)
        self.st3fc3 = nn.Linear(32, 6)

        self.st4fc1 = nn.Linear(13 * 13 * 64, 32)
        self.st4fc2 = nn.Linear(32, 32)
        self.st4fc3 = nn.Linear(32, 6)

        # self.st1fc3.weight.data.zero_()
        # self.st1fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st2fc3.weight.data.zero_()
        self.st2fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st3fc3.weight.data.zero_()
        self.st3fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.st4fc3.weight.data.zero_()
        self.st4fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # def st1(self, x):
    #
    #     xs = x.view(x.size(0), -1)
    #     xs = self.relu(self.st1fc1(xs))
    #     xs = self.relu(self.st1fc2(xs))
    #     theta = self.st1fc3(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #
    #     return x

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
        # x = self.st1(x)
        xin1 = self.hidden1(x)
        xout1 = self.st2(xin1)
        rec1 = self.pl1(xin1, xout1)

        # rec1 = self.maxpool(rec1)

        xin2 = self.hidden2(rec1)
        xout2 = self.st3(xin2)
        rec2 = self.pl2(xin2, xout2)

        xin3 = self.hidden3(rec2)
        xout3 = self.st4(xin3)
        rec3 = self.pl3(xin3, xout3)

        # rec3 = self.maxpool(rec3)

        xin4 = self.hidden4(rec3)
        x = self._features(xin4)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits


class PLinST(nn.Module):
    def __init__(self, r =16):
        super(PLinST, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn= nn.BatchNorm2d(num_features=48)

        self.pl1 = STPL(27, 48, r)
        self.pl2 = STPL(27, 64, r)
        self.pl3 = STPL(13, 128, r)
        # self.pl4 = ResPL(160, r)
        # self.pl5 = ResPL(192, r)
        # self.pl6 = ResPL(192, r)
        # self.pl7 = ResPL(192, r)
        # self.pl8 = ResPL(192, r)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=160),
            # self.pl4,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl5,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl6,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl7,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(num_features=192),
            # self.pl8,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         # m.bias.data.zero_()

    def forward(self, x):
        # x = self.st1(x)
        x = self.hidden1(x)
        x = self.pl1(x)

        x = self.hidden2(x)
        x = self.pl2(x)

        x = self.hidden3(x)
        x = self.pl3(x)

        x = self.hidden4(x)
        x = self._features(x)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits


from st_module import STinSTMulti
class Experim2(nn.Module):
    def __init__(self, r =16):
        super(Experim2, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn= nn.BatchNorm2d(num_features=48)

        self.pl1 = STPL(27, 48, r)
        self.pl2 = STPL(27, 64, r)
        self.pl3 = STPL(13, 128, r)
        # self.pl4 = ResPL(160, r)
        # self.pl5 = ResPL(192, r)
        # self.pl6 = ResPL(192, r)
        # self.pl7 = ResPL(192, r)
        # self.pl8 = ResPL(192, r)
        self.st1 = STinSTMulti(inplanes=48, inshape=54)
        self.st1 = STinSTMulti(inplanes=64, inshape=27)
        self.st1 = STinSTMulti(inplanes=128, inshape=27)
        self.st1 = STinSTMulti(inplanes=160, inshape=13)

        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=160),
            # self.pl4,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=192),
            # self.pl5,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=192),
            # self.pl6,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=192),
            # self.pl7,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Dropout(0.5)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2, bias=True),
            # nn.BatchNorm2d(num_features=192),
            # self.pl8,
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.5)
        )
        hidden9 = nn.Sequential(
            nn.AvgPool2d(3, stride=1)
        )

        self._features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        self._classifier = nn.Sequential(
            hidden9
        )

        self._digit_length = nn.Sequential(nn.Linear(192, 7))
        self._digit1 = nn.Sequential(nn.Linear(192, 11))
        self._digit2 = nn.Sequential(nn.Linear(192, 11))
        self._digit3 = nn.Sequential(nn.Linear(192, 11))
        self._digit4 = nn.Sequential(nn.Linear(192, 11))
        self._digit5 = nn.Sequential(nn.Linear(192, 11))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         # m.bias.data.zero_()

    def forward(self, x):
        # x = self.st1(x)
        x = self.hidden1(x)
        x = self.pl1(x)

        x = self.hidden2(x)
        x = self.pl2(x)

        x = self.hidden3(x)
        x = self.pl3(x)

        x = self.hidden4(x)
        x = self._features(x)
        x = self._classifier(x)
        x = x.view(x.size(0), -1)

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

def stplfcn9(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STPLFCNMulti(r=r)
    return model

def stplagg(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STPLFCNv3(r=r)
    return model

def stplgroupconv(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STPLFCNv4(r=r)
    return model

def stplminus(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STPLminus(r=r)
    return model

def stplinconv(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PLinConv(r=r)
    return model

def respl(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResPLNet(r=r)
    return model

def plinst(r=16):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PLinST(r=r)
    return model

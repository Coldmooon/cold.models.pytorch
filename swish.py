from torch import nn
# import visdom
# import numpy as np
import torch

class SWISH(nn.Module):
    def __init__(self, beta=1, inplace=False, channels=None):
        super(SWISH, self).__init__()
        c = (channels == None) and 1 or int(channels)
        # self.beta = nn.Parameter(torch.ones([1,c,1,1])) # .fill_(beta)
        self.inplace = inplace
        self.sigmoid = nn.Sigmoid()
        # self.count = 0
        # self.vis = visdom.Visdom(port=7777)

    def forward(self, x):

        # if (self.count % 2000 == 0):
        #     if (self.count == 0):
        #         self.vis.line(X=np.array([self.count]), Y=y[0].mean().data.cpu().numpy().reshape(1), win="line2")
        #     else:
        #         self.vis.line(X=np.array([self.count]), Y=y[0].mean().data.cpu().numpy().reshape(1), win="line2", update="append")

        # self.count += 1
        # t = 3.5 * x
        if self.inplace:
            x.mul_(self.sigmoid(3.5 * x))
            return x
        else:
            return x * self.sigmoid(3.5 * x)

class RecSWISH(nn.Module):
    def __init__(self, beta=1, inplace=False, channels=None):
        super(RecSWISH, self).__init__()
        c = (channels == None) and 1 or int(channels)
        # self.beta = nn.Parameter(torch.ones([1,c,1,1])) # .fill_(beta)
        self.inplace = inplace
        self.sigmoid = nn.Sigmoid()
        # self.count = 0
        # self.vis = visdom.Visdom(port=7777)

    def forward(self, x):

        # if (self.count % 2000 == 0):
        #     if (self.count == 0):
        #         self.vis.line(X=np.array([self.count]), Y=y[0].mean().data.cpu().numpy().reshape(1), win="line2")
        #     else:
        #         self.vis.line(X=np.array([self.count]), Y=y[0].mean().data.cpu().numpy().reshape(1), win="line2", update="append")

        # self.count += 1
        if self.inplace:
            x[x<0] = x[x<0] * self.sigmoid(3.5*x[x<0])
            return x
        else:
            t = x.clone() 
            t[t<0] = x[x<0] * self.sigmoid(3.5*x[x<0])
            return t

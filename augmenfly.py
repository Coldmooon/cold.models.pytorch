import torch
from torch import nn
import torch.nn.functional as F
import visdom
import numpy as np

class AUGFLY(nn.Module):
    def __init__(self, inshape, outshape, scale=1., rot=0., trans=0):
        super(AUGFLY, self).__init__()
        self.inshape = inshape
        self.outshape = outshape
        self.vs = visdom.Visdom(port=7666)
        self.scale = scale
        self.rot = rot
        self.trans = trans


    def localization(self, x):
        if isinstance(self.scale, float):
            S = torch.FloatTensor([[self.scale, 0, 0],
                                   [ 0, self.scale,0],
                                   [ 0,      0,    1]]).cuda()
        elif isinstance(self.scale, tuple):
            factor = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1])
            S = torch.FloatTensor([[factor, 0,  0],
                                   [ 0, factor,0],
                                   [ 0,    0,  1]]).cuda()
        else:
            print("ERROR: error type for scale in augmenfly")
            exit(0)

        if isinstance(self.rot, float):
            R = torch.FloatTensor([[torch.cos(self.rot), -torch.sin(self.rot), 0],
                                   [torch.sin(self.rot),  torch.cos(self.rot), 0],
                                   [0, 0, 1]]).cuda()
        elif isinstance(self.rot, tuple):
            factor = torch.FloatTensor(1).uniform_(self.rot[0], self.rot[1])
            R = torch.FloatTensor([[torch.cos(factor), -torch.sin(factor), 0],
                                   [torch.sin(factor),  torch.cos(factor), 0],
                                   [0, 0, 1]]).cuda()
        else:
            print("ERROR: error type for rotation in augmenfly")
            exit(0)

        theta = torch.mm(S, R)[0:2,:]

        return theta.expand(x.size(0), 2, 3)

    def st(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size([x.size(0), x.size(1), self.outshape, self.outshape]))
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        out = self.st(x)
        # out = x + xs
        # out = (out - torch.mean(out))/torch.std(out)
        # self.vs.images(x.data.cpu().numpy(), win=1, opts={"title": "The original input images"})
        # self.vs.images(xs.data.cpu().numpy(), win=2, opts={"title": "The warped input images"})
        # img = out.data.cpu().numpy() * 255
        # self.vs.images(img.astype(np.uint8), win=3, opts={"title": "Merged image"})

        return out

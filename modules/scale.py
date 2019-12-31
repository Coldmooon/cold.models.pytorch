from torch import nn
import torch
import visdom
import numpy as np

class ScaleLayer(nn.Module):

   def __init__(self, channels):
       super().__init__()

       self.scale = nn.Parameter(torch.ones(1, channels, 1, 1)*20, requires_grad=True)
       self.bias  = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

       self.count = 0
       self.vis = visdom.Visdom(port=7777)

   def forward(self, input):

       if (self.count % 100 == 0):
           if (self.count == 0):
               self.vis.line(X=np.array([self.count]), Y=self.scale.mean().data.cpu().numpy().reshape(1), win="weight", opts={ "title": "scale.weight" })
               self.vis.line(X=np.array([self.count]), Y=self.bias.mean().data.cpu().numpy().reshape(1), win="bias", opts={ "title": "scale.bias" })
           else:
               self.vis.line(X=np.array([self.count]), Y=self.scale.mean().data.cpu().numpy().reshape(1), win="weight",
                             update="append")
               self.vis.line(X=np.array([self.count]), Y=self.bias.mean().data.cpu().numpy().reshape(1),
                             win="bias",
                             update="append")

       self.count += 1

       return input * self.scale + self.bias

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hourglass(nn.Module):
    def __init__(self):
        super(Hourglass, self).__init__()
        # Down sampling
        self.down_conv1 = nn.Conv2d(3, 3, 3)
        self.down_conv2 = nn.Conv2d(3, 3, 3)
        self.down_conv3 = nn.Conv2d(3, 3, 3)
        self.down_conv4 = nn.Conv2d(3, 3, 3)
        self.down_conv5 = nn.Conv2d(3, 3, 3)
        
        # Bottle neck
        self.bottle_conv = nn.Conv2d(3, 3, 3)
        
        # Up sampling
        self.up_conv1 = nn.Conv2d(3, 3, 3)
        self.up_conv2 = nn.Conv2d(3, 3, 3)
        self.up_conv3 = nn.Conv2d(3, 3, 3)
        self.up_conv4 = nn.Conv2d(3, 3, 3)
        self.up_conv5 = nn.Conv2d(3, 3, 3)
        
        # Branches
        self.branch_conv1 = nn.Conv2d(3, 3, 3)
        self.branch_conv2 = nn.Conv2d(3, 3, 3)
        self.branch_conv3 = nn.Conv2d(3, 3, 3)
        self.branch_conv4 = nn.Conv2d(3, 3, 3)
        self.branch_conv5 = nn.Conv2d(3, 3, 3)
        
        # Other
        self.up_sample = nn.Upsample(mode = "nearest")
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Down sampling
        x_down_conv1 = F.relu(self.down_colv1(x))
        x_pool1 = self.pool(x_down_conv1)
        x_down_conv2 = F.relu(self.down_colv1(x_pool1))
        x_pool2 = self.pool(x_down_conv2)
        x_down_conv3 = F.relu(self.down_colv1(x_pool2))
        x_pool3 = self.pool(x_down_conv3)
        x_down_conv4 = F.relu(self.down_colv1(x_pool3))
        x_pool4 = self.pool(x_down_conv4)
        x_down_conv5 = F.relu(self.down_colv1(x_pool4))
        x_pool5 = self.pool(x_down_conv5)
        
        # Bottle neck
        x_bottle_conv = self.bottle_cov(x_pool5)
        
        # Branches
        x_branch_conv1 = F.relu(self.down_colv1(x_down_conv1))
        x_branch_conv2 = F.relu(self.down_colv1(x_down_conv2))
        x_branch_conv3 = F.relu(self.down_colv1(x_down_conv3))
        x_branch_conv4 = F.relu(self.down_colv1(x_down_conv4))
        x_branch_conv5 = F.relu(self.down_colv1(x_down_conv5))
        
        # Up sampling
        

class SHG(nn.Module):
    pass

net = SHG()
net.cuda()
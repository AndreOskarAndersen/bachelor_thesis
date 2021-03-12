import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F
import torch.cuda
import torch.optim as optim
from tqdm.notebook import tqdm

class Residual(nn.Module):
    def __init__(self, in_channels = 256):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 128, kernel_size = 1)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 1)
        self.branch_cov = nn.Conv2d(in_channels = in_channels, out_channels = 256, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        branch = self.branch_cov(x)

        x = self.conv1(x)
        x = relu(x)
        
        x = self.conv2(x)
        x = relu(x)
        
        x = self.conv3(x)
        x = relu(x)
        
        return x + branch

class Hourglass(nn.Module):
    def __init__(self, num_layers = 4, num_bottlenecks = 3):
        super(Hourglass, self).__init__()
        self.num_layers = num_layers
        self.num_bottlenecks = num_bottlenecks
        
        self.encoder_max_poolings = [] # downsampling
        self.encoder_residuals = [] # downsampling
        self.decoder_upsamplings = [] # upsampling
        self.decoder_residuals = [] # upsampling
        self.branch_cov = []
        self.bottlenecks = []
        
        for i in range(self.num_layers):
            self.encoder_max_poolings.append(nn.MaxPool2d(2, stride = 2))
            self.encoder_residuals.append(Residual())
            self.decoder_upsamplings.append(nn.Upsample(scale_factor = 2))
            if (i != self.num_layers - 1): # Last layer does not end on residual
                self.decoder_residuals.append(Residual())
            self.branch_cov.append(Residual())
            
        self.encoder_max_poolings = nn.ModuleList(self.encoder_max_poolings)
        self.encoder_residuals = nn.ModuleList(self.encoder_residuals)
        self.decoder_upsamplings = nn.ModuleList(self.decoder_upsamplings)
        self.decoder_residuals = nn.ModuleList(self.decoder_residuals)
        self.branch_cov = nn.ModuleList(self.branch_cov)
        self.bottlenecks = nn.ModuleList([Residual() for _ in range(self.num_bottlenecks)])
        
    def forward(self, x):
        self.branch = []
        
        # Encoding
        for i in range(self.num_layers):
            self.branch.append(self.branch_cov[i](x))
            x = self.encoder_max_poolings[i](x)
            x = self.encoder_residuals[i](x)
            
        # Bottleneck
        for i in range(self.num_bottlenecks):
            x = self.bottlenecks[i](x)
            
        # Decode
        for i in range(self.num_layers - 1):
            x = self.decoder_upsamplings[i](x)
            x = x + self.branch[-1 * i - 1] # branches are stored backwards
            x = self.decoder_residuals[i](x)
            
        x = self.decoder_upsamplings[-1](x) # Last layer does not end on residual
            
        return x + self.branch[0]

class SHG(nn.Module):
    def __init__(self, num_hourglasses = 2, num_layers = 4, num_bottlenecks = 2):
        super(SHG, self).__init__()
        self.num_hourglasses = num_hourglasses
        self.num_layers = num_layers
        
        # Before the first hourglass
        self.pre_conv1 = nn.Conv2d(3, 256, stride=2, kernel_size = 7, padding = 3)
        self.pre_residual1 = Residual()
        self.pre_max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pre_residual2 = Residual()
        self.pre_residual3 = Residual()
        
        # After each hourglass
        self.post_conv_1 = []
        self.post_conv_2 = []
        self.post_pred_conv = []
        self.input_branch = None # Clone of the input to hourglass, used for branch (see intermediate supervision process)
        
        # Hourglasses
        self.hourglasses = nn.ModuleList([Hourglass(num_layers=num_layers, num_bottlenecks=num_bottlenecks) for _ in range(self.num_hourglasses)])
        
        for _ in range(self.num_hourglasses - 1):
            self.post_conv_1.append(nn.Conv2d(256, 17, kernel_size = 1))
            self.post_conv_2.append(nn.Conv2d(17, 256, kernel_size = 1))
            self.post_pred_conv.append(nn.Conv2d(17, 256, kernel_size = 1))
          
        self.post_conv_1 = nn.ModuleList(self.post_conv_1)
        self.post_conv_2 = nn.ModuleList(self.post_conv_2)
        self.post_pred_conv = nn.ModuleList(self.post_pred_conv)
        
        self.last_conv_1 = nn.Conv2d(256, 17, kernel_size = 1)
        self.last_conv_2 = nn.Conv2d(17, 17, kernel_size = 1)
            
    def forward(self, x):
        self.pred = []
        x = self.pre_conv1(x)
        x = relu(x)
        x = self.pre_residual1(x)
        x = self.pre_max_pool(x)
        x = self.pre_residual2(x)
        x = self.pre_residual3(x)
        
        for i in range(self.num_hourglasses - 1):
            self.input_branch = torch.clone(x)
            x = self.hourglasses[i](x)
            x = self.post_conv_1[i](x) 
            x = relu(x)
            self.pred.append(torch.clone(x))
            x = self.post_conv_2[i](x)
            x = relu(x)
            x = self.input_branch + x + relu(self.post_pred_conv[i](self.pred[i]))
                    
        x = self.hourglasses[-1](x)

        x = self.last_conv_1(x)
        x = relu(x)
        x = self.last_conv_2(x)
        x = relu(x)
        self.pred.append(x)

        #for i in range(len(self.pred)):
        #  self.pred[i] = self.pred[i].cpu().detach().numpy()
        
        return self.pred[-1]
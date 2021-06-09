import torch.nn as nn
from torch.nn.functional import relu

class SHG_AE(nn.Module):
    def __init__(self, SHG_model, AE_model):
        super(SHG_AE, self).__init__()
        self.SHG_model = SHG_model
        self.AE_model = AE_model
        
    def forward(self, x, add_noise = True):
        x = self.SHG_prework(x)
        x = self.encode(x, add_noise=add_noise)
        x = self.decode(x)
        x = self.SHG_postwork(x)
        
        return x
    
    def SHG_prework(self, x):
        x = self.SHG_model.pre_conv1(x)
        x = self.SHG_model.bn_pre(x)
        x = relu(x)
        x = self.SHG_model.pre_residual1(x)
        x = self.SHG_model.pre_max_pool(x)
        x = self.SHG_model.pre_residual2(x)
        x = self.SHG_model.pre_residual3(x)
        
        return x
    
    def encode(self, x, add_noise):
        x = self.SHG_model.hourglasses[-1].encode(x)
        x, _ = self.SHG_model.hourglasses[-1].bottleneck(x)
        x = self.AE_model.encode(x, add_noise)
        
        return x
        
    def decode(self, x):
        x = self.AE_model.decode(x)
        x = self.SHG_model.hourglasses[-1].decode(x)
        
        return x
    
    def SHG_postwork(self, x):
        x = self.SHG_model.last_res(x)
        x = self.SHG_model.last_conv_1(x)
        x = self.SHG_model.bn_last(x)
        x = relu(x)
        x = self.SHG_model.last_conv_2(x)
        
        return x
        
    
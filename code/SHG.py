import torch.nn as nn
from torch.nn.functional import relu

class Residual(nn.Module):
    def __init__(self, in_channels = 256):
        super(Residual, self).__init__()
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 128, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 1)
        self.branch_cov = nn.Conv2d(in_channels = in_channels, out_channels = 256, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        branch = self.branch_cov(x)

        x = self.input_bn(x)
        x = relu(x)
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = relu(x)
        x = self.conv2(x)
        
        x = self.bn2(x)
        x = relu(x)
        x = self.conv3(x)
        
        return x + branch

class Hourglass(nn.Module):
    def __init__(self, num_layers = 4, num_bottlenecks = 3, use_skip_connections = True):
        super(Hourglass, self).__init__()
        self.num_layers = num_layers
        self.num_bottlenecks = num_bottlenecks
        self.use_skip_connections = use_skip_connections
        
        self.encoder_max_poolings = [] # downsampling
        self.encoder_residuals = [] # downsampling
        self.decoder_upsamplings = [] # upsampling
        self.decoder_residuals = [] # upsampling
        self.branch_cov = []
        self.bottlenecks = []
        self.branch = []
        
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
        
    def encode(self, x):
        if (self.use_skip_connections): # XAI: we only use the skip connections sometimes
            self.branch = []
        
        for i in range(self.num_layers):
            if (self.use_skip_connections): # XAI: we only use the skip connections sometimes
                self.branch.append(self.branch_cov[i](x))
                
            x = self.encoder_max_poolings[i](x)
            x = self.encoder_residuals[i](x)
            
        return x
    
    def bottleneck(self, x):
        bottleneck_res = []
        
        for i in range(self.num_bottlenecks):
            x = self.bottlenecks[i](x)
            bottleneck_res.append(x)
            
        return x, bottleneck_res
    
    def decode(self, x):
        for i in range(self.num_layers - 1):
            x = self.decoder_upsamplings[i](x)
            
            if (self.use_skip_connections): # XAI: we only use the skip connections sometimes
                x = x + self.branch[-1 * i - 1] # branches are stored backwards
                
            x = self.decoder_residuals[i](x)
            
        x = self.decoder_upsamplings[-1](x) # Last layer does not end on residual
        
        if (self.use_skip_connections):
            x = x + self.branch[0]
            
        return x
        
    def forward(self, x):
        x = self.encode(x)
        x, bottleneck_res = self.bottleneck(x)
        x = self.decode(x)
        
        return x, bottleneck_res

class SHG(nn.Module):
    def __init__(self, num_hourglasses, num_layers = 4, num_bottlenecks = 3, use_skip_connections = True):
        super(SHG, self).__init__()
        self.num_hourglasses = num_hourglasses
        self.num_layers = num_layers
        
        # Before the first hourglass
        self.pre_conv1 = nn.Conv2d(3, 256, stride=2, kernel_size = 7, padding = 3)
        self.bn_pre = nn.BatchNorm2d(256)
        self.pre_residual1 = Residual()
        self.pre_max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pre_residual2 = Residual()
        self.pre_residual3 = Residual()
        
        # After each hourglass
        self.post_res = []
        self.post_conv_1 = []
        self.post_conv_2 = []
        self.pre_pred_conv = []
        self.post_pred_conv = []
        self.input_branch = None # Clone of the input to hourglass, used for branch (see intermediate supervision process)
        self.pred_branch = None
        
        # Hourglasses
        self.hourglasses = nn.ModuleList([Hourglass(num_layers=num_layers, num_bottlenecks=num_bottlenecks, use_skip_connections=use_skip_connections) for _ in range(self.num_hourglasses)])
        
        for _ in range(self.num_hourglasses - 1):
            self.post_res.append(Residual())
            self.post_conv_1.append(nn.Conv2d(256, 256, kernel_size = 1))
            self.post_conv_2.append(nn.Conv2d(256, 256, kernel_size = 1))
            self.pre_pred_conv.append(nn.Conv2d(256, 17, kernel_size = 1))
            self.post_pred_conv.append(nn.Conv2d(17, 256, kernel_size = 1))
          
        self.post_res = nn.ModuleList(self.post_res)
        self.post_conv_1 = nn.ModuleList(self.post_conv_1)
        self.post_conv_2 = nn.ModuleList(self.post_conv_2)
        self.pre_pred_conv = nn.ModuleList(self.pre_pred_conv)
        self.post_pred_conv = nn.ModuleList(self.post_pred_conv)
        
        # Before the last output
        self.last_res = Residual()
        self.last_conv_1 = nn.Conv2d(256, 17, kernel_size = 1)
        self.bn_last = nn.BatchNorm2d(17)
        self.last_conv_2 = nn.Conv2d(17, 17, kernel_size = 1)
        
        # Initialize weights
        self.init_params()
        
    def init_params(self):
        for p in self.parameters():
            if (len(p.shape) > 1): # cannot init batchnorms. 
                nn.init.xavier_normal_(p)
    
    def decode(self, x):
        """ Given a sample from the latent-space (x) of the last hourglass, return the corresponding output of the whole network"""                
                    
        x = self.hourglasses[-1].decode(x)
        x = self.last_res(x)
        x = self.last_conv_1(x)
        x = self.bn_last(x)
        x = relu(x)
        x = self.last_conv_2(x)
            
        return x
    
    def forward(self, x):
        x = self.pre_conv1(x)
        x = self.bn_pre(x)
        x = relu(x)
        x = self.pre_residual1(x)
        x = self.pre_max_pool(x)
        x = self.pre_residual2(x)
        x = self.pre_residual3(x)
            
        for i in range(self.num_hourglasses - 1):
            self.input_branch = x
            x = self.hourglasses[i](x)
            x = self.post_res[i](x)
            x = self.post_conv_1[i](x)
            x = relu(x)
            self.pred_branch = x
            self.pred_branch = self.pre_pred_conv[i](self.pred_branch)
            self.pred_branch  = relu(self.pred_branch)
            self.pred_branch = self.post_pred_conv[i](self.pred_branch)
            self.pred_branch  = relu(self.pred_branch)
            x = self.post_conv_2[i](x)
            x = relu(x)
            x = x + self.input_branch + self.pred_branch
                  
        x, bottleneck_res = self.hourglasses[-1](x)
        
        x = self.last_res(x)
        x = self.last_conv_1(x)
        x = self.bn_last(x)
        x = relu(x)
        x = self.last_conv_2(x)
        
        return x, bottleneck_res
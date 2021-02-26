import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

class Residual(nn.Module):
    def __init__(self, in_channels = 256):
        super(Residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 128, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 1)
        self.branch_cov = nn.Conv2d(in_channels = in_channels, out_channels = 256, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        branch = self.branch_cov(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3(x)
        
        x = x + branch
        
        return x
      
class Hourglass(nn.Module):
    def __init__(self, num_layers = 4, num_bottlenecks = 3):
        super(Hourglass, self).__init__()
        self.num_layers = num_layers
        self.num_bottlenecks = num_bottlenecks
        
        self.encoder_max_poolings = [] # downsampling
        self.encoder_residuals = [] # downsampling
        self.decoder_upsamplings = [] # upsampling
        self.decoder_residuals = [] # upsampling
        self.branch = []
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
        x = x + self.branch[0]
            
        return x
            
class SHG(nn.Module):
    def __init__(self, num_hourglasses = 2, num_layers = 4, num_bottlenecks = 2):
        super(SHG, self).__init__()
        self.num_hourglasses = num_hourglasses
        self.num_layers = num_layers
        
        # Before any hourglass
        self.pre_conv1 = nn.Conv2d(3, 3, stride=2, kernel_size = 7)
        self.pre_residual1 = Residual(in_channels = 3)
        self.pre_max_pool = nn.MaxPool2d(kernel_size = 62, stride = 1)
        self.pre_residual2 = Residual()
        self.pre_residual3 = Residual()
        
        # After each hourglass
        self.post_conv_1 = []
        self.post_conv_2 = []
        self.post_blue_box_conv = []
        self.blue_box = []
        self.input_branch = None # Clone of the input to hourglass, used for branch (see intermediate supervision process)
        
        # Hourglasses
        self.hourglasses = nn.ModuleList([Hourglass(num_layers=num_layers, num_bottlenecks=num_bottlenecks) for _ in range(self.num_hourglasses)])
        
        for _ in range(self.num_hourglasses - 1):
            self.post_conv_1.append(nn.Conv2d(256, 17, kernel_size = 1))
            self.post_conv_2.append(nn.Conv2d(17, 256, kernel_size = 1))
            self.post_blue_box_conv.append(nn.Conv2d(17, 256, kernel_size = 1))
          
        self.post_conv_1 = nn.ModuleList(self.post_conv_1)
        self.post_conv_2 = nn.ModuleList(self.post_conv_2)
        self.post_blue_box_conv = nn.ModuleList(self.post_blue_box_conv)
            
    def forward(self, x):        
        x = self.pre_conv1(x)
        x = self.pre_residual1(x)
        x = self.pre_max_pool(x)
        x = self.pre_residual2(x)
        x = self.pre_residual3(x)
        
        for i in range(self.num_hourglasses - 1):
            self.input_branch = torch.clone(x)
            x = self.hourglasses[i](x) # VIRKER IKKE; one of the variables needed for gradient computation has been modified by an inplace operation.
            x = self.post_conv_1[i](x) 
            self.blue_box.append(torch.clone(x))
            x = self.post_conv_2[i](x)
            x = self.input_branch + x + self.post_blue_box_conv[i](self.blue_box[i])
                    
        x = self.hourglasses[-1](x)
        x = nn.Conv2d(256, 17, kernel_size = 1)(x)
        x = nn.Conv2d(17, 17, kernel_size = 1)(x)
        self.blue_box.append(x)
        
        return x

USE_GPU = False
LEARNING_RATE = 2.5e-4
NUM_EPOCHS = 10

if USE_GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    model = SHG().to(device)
    x = torch.randn(64, 3, 256, 256).cuda()
    print(model(x).shape)
else:
    model = SHG()
    X = torch.randn(100, 3, 256, 256)
    all_heatmaps = torch.randn(100, 17, 64, 64)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr = LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        for x, heatmaps in zip(X, all_heatmaps):
            x = torch.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
            heatmaps = torch.reshape(heatmaps, (1, heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]))
            
            model(x)
            
            losses = []
            for i in range(model.num_hourglasses):
                losses.append(criterion(model.blue_box[i], heatmaps))
                
            losses = torch.stack(losses)
            
            for loss in reversed(losses):
                optimizer.zero_grad()
                loss.backward() # Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph = True
                optimizer.step()
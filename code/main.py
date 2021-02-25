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
        
        for _ in range(self.num_layers):
            self.encoder_max_poolings.append(nn.MaxPool2d(2, stride = 2))
            self.encoder_residuals.append(Residual())
            self.decoder_upsamplings.append(nn.Upsample(scale_factor = 2))
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
            
    def forward(self, x, print_loss = False, true_heatmaps = None, criterion = None):
        self.loss = []
        self.print_loss = print_loss
        self.true_heatmaps = true_heatmaps
        self.criterion = criterion
        """
        x = self.pre_conv1(x)
        x = self.pre_residual1(x)
        x = self.pre_max_pool(x)
        x = self.pre_residual2(x)
        x = self.pre_residual3(x)
        
        for i in range(self.num_hourglasses - 1):
            self.input_branch = torch.clone(x)
            x = self.hourglasses[i](x)
            x = self.post_conv_1[i](x)
            self.blue_box.append(torch.clone(x))
            x = self.post_conv_2[i](x)
            x = self.input_branch + x + self.post_blue_box_conv[i](self.blue_box[i])
            
            # loss
            if (self.true_heatmaps is not None and self.criterion is not None):
                self.loss.append(self.criterion(self.blue_box[-1], self.true_heatmaps))
                if (self.print_loss):
                    print("loss at hourglass {}: {}".format(i, self.loss[-1]))
                    
        x = self.hourglasses[-1](x)
        x = nn.Conv2d(256, 17, kernel_size = 1)(x)
        x = nn.Conv2d(17, 17, kernel_size = 1)(x)
        
        # loss for last layer
        if (self.true_heatmaps is not None and self.criterion is not None):
            self.loss.append(self.criterion(x, self.true_heatmaps))
            if (self.print_loss):
                print("loss at hourglass {}: {}".format(self.num_hourglasses - 1, self.loss[-1]))
        
        return x
        """
                    
        pre_conv1 = self.pre_conv1(x)
        pre_residual1 = self.pre_residual1(pre_conv1)
        pre_max_pool = self.pre_max_pool(pre_residual1)
        pre_residual2 = self.pre_residual2(pre_max_pool)
        pre_residual3 = self.pre_residual3(pre_residual2)
        inputs = [pre_residual3]
        hourglass_res = []
        post_conv_1_res = []
        post_conv_2_res = []
        
        for i in range(self.num_hourglasses - 1):
            hourglass_res.append(self.hourglasses[i](inputs[-1]))
            post_conv_1_res.append(self.post_conv_1[i](hourglass_res[-1]))
            post_conv_2_res.append(self.post_conv_2[i](post_conv_1_res[-1]))
            inputs.append(inputs[-1] + post_conv_2_res[-1] + self.post_blue_box_conv[i](post_conv_1_res[-1]))
            
            # loss
            if (self.true_heatmaps is not None and self.criterion is not None):
                self.loss.append(self.criterion(post_conv_1_res[-1], self.true_heatmaps))
                if (self.print_loss):
                    print("loss at hourglass {}: {}".format(i, self.loss[-1]))
            
        hourglass_res.append(torch.clone(self.hourglasses[-1](inputs[-1])))
        out_1 = nn.Conv2d(256, 17, kernel_size = 1)(hourglass_res[-1])
        out_2 = nn.Conv2d(17, 17, kernel_size = 1)(out_1)
        
        # loss for last layer
        if (self.true_heatmaps is not None and self.criterion is not None):
            self.loss.append(self.criterion(out_2, self.true_heatmaps))
            if (self.print_loss):
                print("loss at hourglass {}: {}".format(self.num_hourglasses - 1, self.loss[-1]))

        return out_2

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
    heatmaps = torch.randn(100, 17, 64, 64)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr = LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        for x, heatmap in zip(X, heatmaps):
            x = torch.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
            heatmap = torch.reshape(heatmap, (1, heatmap.shape[0], heatmap.shape[1], heatmap.shape[2]))
            model(x, true_heatmaps = heatmap, criterion = criterion)
            
            for loss in reversed(model.loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
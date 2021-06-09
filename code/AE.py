from matplotlib.pyplot import xticks
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

        self.encoder_linear = nn.Sequential(
            nn.Linear(1024, 50),
            nn.ReLU(inplace = True)
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(inplace = True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size = 3, padding = 1), # Works as a conv2d, as it only changes the amount of filters used
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 192, kernel_size = 3, padding = 1), # Works as a conv2d, as it only changes the amount of filters used
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(192, 256, kernel_size = 3, padding = 1), # Works as a conv2d, as it only changes the amount of filters used
            nn.ReLU(inplace = True)
        )

        # Initialize weights
        self.init_params()
        
    def init_params(self):
        for p in self.parameters():
            if (len(p.shape) > 1):
                nn.init.xavier_normal_(p)

    def encode(self, X, add_noise):
        if (add_noise):
            X = self.add_noise(X)
        
        X = self.encoder(X)
        X = X.view(X.shape[0], -1)

        X = self.encoder_linear(X)

        return X

    def decode(self, X):
        X = self.decoder_linear(X)
        X = X.view(X.shape[0], 64, 4, 4)
        
        X = self.decoder(X)
        
        return X

    def add_noise(self, X, fac = 0.1):
        noise = torch.randn_like(X) * fac * X
        return X + noise

    def forward(self, X, add_noise = True):        
        X = self.encode(X, add_noise)
        X = self.decode(X)
        
        return X
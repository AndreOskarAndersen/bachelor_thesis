import torch
import torch.nn as nn

class CONV_AE(nn.Module):
    def __init__(self):
        super(CONV_AE, self).__init__()
        # Måske gøre brug af batch normalization nogle steder?
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True) # """  MANGLER OGSÅ LINEAR
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace = True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(192, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

        # Initialize weights
        self.init_params()
        
    def init_params(self):
        for p in self.parameters():
            if (len(p.shape) > 1):
                nn.init.xavier_normal_(p)

    def encode(self, X):
        X = self.encoder(X)
        X = X.view(X.shape[0], -1)

        X = self.bottleneck(X)
        X = X.view(X.shape[0], 64, 4, 4)

        return X

    def decode(self, X):
        return self.decoder(X)

    def add_noise(self, X, fac = 1):
        noise = torch.randn_like(X) * fac
        return X + noise

    def forward(self, X, add_noise = True):

        if (add_noise):
            X = self.add_noise(X)
        
        X_encoded = self.encode(X)

        X_decoded = self.decode(X_encoded)
        return X_encoded, X_decoded
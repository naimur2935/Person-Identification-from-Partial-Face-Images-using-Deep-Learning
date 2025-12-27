import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU())

        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.ReLU())
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        d4 = self.dec4(torch.cat([d3, e1], dim=1))

        return d4

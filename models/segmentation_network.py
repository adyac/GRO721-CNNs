import torch
import torch.nn as nn

class Unet_model(nn.Module):

    def __init__(self):
        super(Unet_model, self).__init__()

        self.encodeur1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encodeur2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encodeur3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )

        self.transposeconv1 = nn.ConvTranspose2d(256, 128, 2, 2,output_padding=1)
        self.decodeur1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU()
        )

        self.transposeconv2 = nn.ConvTranspose2d(128, 64, 2, 2)

        self.decodeur2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )

        self.transposeconv3 = nn.ConvTranspose2d(64, 32, 2, 2,output_padding=1)
        self.decodeur3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )

        self.convfinal = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        x1 = self.encodeur1(x)
        x2 = self.pool1(x1)

        x3 = self.encodeur2(x2)
        x4 = self.pool2(x3)

        x5 = self.encodeur3(x4)
        x6 = self.pool3(x5)

        x7 = self.bottleneck(x6)

        u1 = self.transposeconv1(x7)
        cat1 = torch.cat([u1, x5], dim=1)
        d1 = self.decodeur1(cat1)

        u2 = self.transposeconv2(d1)
        cat2 = torch.cat([u2, x3], dim=1)
        d2 = self.decodeur2(cat2)

        u3 = self.transposeconv3(d2)
        cat3 = torch.cat([u3, x1], dim=1)
        d3 = self.decodeur3(cat3)

        out = self.convfinal(d3)
        return out

def build_segmentation_model():
    return Unet_model()


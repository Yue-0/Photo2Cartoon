import networks.nn as nn
from networks.fcn import FCN
from networks.unet import UNet


__all__ = ["Generator", "Discriminator"]


class Generator(nn.Layer):
    def __init__(self, channels):
        super(Generator, self).__init__()
        self.person2cartoon = nn.Sequential(
            UNet(channels),
            nn.ReLU(),
            nn.TransposedConv2D(channels >> 2, 3, 4, 2, 1),
            nn.TanH()
        )

    def forward(self, person):
        return self.person2cartoon(person)


class Discriminator(nn.Layer):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.concat = nn.Concat(1)
        self.classifier = nn.Sequential(
            nn.Conv2D(6, channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            FCN(channels),
            nn.Conv2D(channels << 3, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, *images):
        return self.classifier(self.concat(*images))

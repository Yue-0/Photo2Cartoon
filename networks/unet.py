import networks.nn as nn


__all__ = ["UNet"]


class DownSampling(nn.ConvBlock):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__(
            nn.Conv2D, in_channels, out_channels, nn.LeakyReLU, True,
            True, False, 0.2, kernel_size=4, stride=2, padding=1, bias=False
        )


class UpSampling(nn.ConvBlock):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(UpSampling, self).__init__(
            nn.TransposedConv2D, in_channels, out_channels, nn.ReLU, True,
            True, dropout, kernel_size=4, stride=2, padding=1, bias=False
        )


class UNet(nn.Layer):
    def __init__(self, channels):
        super(UNet, self).__init__()
        self.encoder1 = nn.Conv2D(3, channels >> 3, 4, 2, 1)
        self.encoder2 = DownSampling(channels >> 3, channels >> 2)
        self.encoder3 = DownSampling(channels >> 2, channels >> 1)
        self.encoder4 = DownSampling(channels >> 1, channels >> 0)
        self.encoder5 = DownSampling(channels >> 0, channels >> 0)
        self.encoder6 = DownSampling(channels >> 0, channels >> 0)
        self.encoder7 = DownSampling(channels >> 0, channels >> 0)
        self.center_block = DownSampling(channels, channels)
        self.decoder7 = UpSampling(channels << 0, channels << 0, True)
        self.decoder6 = UpSampling(channels << 1, channels << 0, True)
        self.decoder5 = UpSampling(channels << 1, channels << 0, True)
        self.decoder4 = UpSampling(channels << 1, channels << 0)
        self.decoder3 = UpSampling(channels << 1, channels >> 1)
        self.decoder2 = UpSampling(channels >> 0, channels >> 2)
        self.decoder1 = UpSampling(channels >> 1, channels >> 3)

    def forward(self, image):
        image1 = self.encoder1(image)
        image2 = self.encoder2(image1)
        image3 = self.encoder3(image2)
        image4 = self.encoder4(image3)
        image5 = self.encoder5(image4)
        image6 = self.encoder6(image5)
        image7 = self.encoder7(image6)
        image = self.center_block(image7)
        image = self.decoder7(image, image7)
        image = self.decoder6(image, image6)
        image = self.decoder5(image, image5)
        image = self.decoder4(image, image4)
        image = self.decoder3(image, image3)
        image = self.decoder2(image, image2)
        image = self.decoder1(image, image1)
        return image

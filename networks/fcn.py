import networks.nn as nn


__all__ = ["FCN"]


class CBL(nn.ConvBlock):
    def __init__(self, in_channels, out_channels, s=2):
        super(CBL, self).__init__(
            nn.Conv2D, in_channels, out_channels, nn.LeakyReLU, False,
            True, False, 0.2, kernel_size=4, stride=s, padding=1, bias=False
        )


class FCN(nn.Layer):
    def __init__(self, channels):
        super(FCN, self).__init__()
        self.conv = nn.Sequential(
            CBL(channels << 0, channels << 1),
            CBL(channels << 1, channels << 2),
            CBL(channels << 2, channels << 3, 1),
        )

    def forward(self, image):
        return self.conv(image)

from torch import nn


class ConvINReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, negative_slope=0.2):
        padding = (kernel_size - 1) // 2
        super(ConvINReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(negative_slope, inplace=True),
        )


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ConvINReLU(64, 128, 4, stride=2),
            ConvINReLU(128, 256, 4, stride=2),
            ConvINReLU(256, 512, 4, stride=1),
            nn.Conv2d(512, 1, 4, 1),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

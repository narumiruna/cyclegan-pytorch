from torch import nn


class ConvINLeakyReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, negative_slope=0.2):
        padding = (kernel_size - 1) // 2
        super(ConvINLeakyReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(negative_slope, inplace=True),
        )


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        layers = [
            ConvINLeakyReLU(3, 64, 4, stride=2),
            ConvINLeakyReLU(64, 128, 4, stride=2),
            ConvINLeakyReLU(128, 256, 4, stride=2),
            ConvINLeakyReLU(256, 512, 4, stride=1),
            nn.Conv2d(512, 1, 4, 1),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

    def numel(self):
        return sum(p.numel() for p in self.parameters())

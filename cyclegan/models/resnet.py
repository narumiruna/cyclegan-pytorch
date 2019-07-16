from torch import nn


class ConvINLeakyReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, negative_slope=0.2):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=0),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]
        super(ConvINLeakyReLU, self).__init__(*layers)


class DeconvINLeakyReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, negative_slope=0.2):
        layers = [
            nn.ConvTranspose2d(in_planes, out_planes, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]
        super(DeconvINLeakyReLU, self).__init__(*layers)


class ResnetGenerator(nn.Sequential):

    def __init__(self):
        layers = [
            ConvINLeakyReLU(3, 64, 7, 1),
            ConvINLeakyReLU(64, 128, 3, stride=2),
            ConvINLeakyReLU(128, 256, 3, stride=2),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            DeconvINLeakyReLU(256, 128),
            DeconvINLeakyReLU(128, 64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]
        super(ResnetGenerator, self).__init__(*layers)


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, negative_slope=0.2):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.features = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size, stride=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size, stride=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.features(x)

    def numel(self):
        return sum(p.numel() for p in self.parameters())

from torch import nn


class ConvINReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
        super(ConvINReLU, self).__init__(*layers)


class DeconvINLeakyReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes):
        layers = [
            nn.ConvTranspose2d(in_planes, out_planes, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
        super(DeconvINLeakyReLU, self).__init__(*layers)


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.features = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, stride=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, stride=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.features(x)


class ResnetGenerator(nn.Sequential):

    def __init__(self, num_resblocks=9):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            ConvINReLU(64, 128, 3, stride=2),
            ConvINReLU(128, 256, 3, stride=2),
        ]

        for _ in range(num_resblocks):
            layers += [ResidualBlock(256)]

        layers += [
            DeconvINLeakyReLU(256, 128),
            DeconvINLeakyReLU(128, 64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]
        super(ResnetGenerator, self).__init__(*layers)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

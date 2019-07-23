from torch import nn


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups),
            nn.InstanceNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class DeconvINReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes):
        super(DeconvINReLU, self).__init__(
            nn.ConvTranspose2d(in_planes, out_planes, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobilenetGenerator(nn.Sequential):

    def __init__(self, num_resblocks=9):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            InvertedResidual(64, 128, stride=2),
            InvertedResidual(128, 256, stride=2),
        ]

        for _ in range(num_resblocks):
            layers += [InvertedResidual(256, 256, stride=1)]

        layers += [
            DeconvINReLU(256, 128),
            DeconvINReLU(128, 64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]
        super(MobilenetGenerator, self).__init__(*layers)

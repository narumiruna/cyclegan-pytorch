import torch
from torch import nn


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class DeconvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes):
        super(DeconvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
            nn.ConvTranspose2d(out_planes, out_planes, 3, 2, 1, 1, groups=out_planes, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, expand_ratio, kernel_size, stride, reduction_ratio=4):
        super(MBConvBlock, self).__init__()
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientGenerator(nn.Sequential):

    def __init__(self):

        features = [
            ConvBNReLU(3, 32, 3, stride=2),
            # [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            MBConvBlock(32, 16, expand_ratio=1, kernel_size=3, stride=1),
            # [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            MBConvBlock(16, 24, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(24, 24, expand_ratio=6, kernel_size=3, stride=1),
            # [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            MBConvBlock(24, 40, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock(40, 40, expand_ratio=6, kernel_size=5, stride=1),
            # [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            MBConvBlock(40, 80, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1),
            # [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            MBConvBlock(80, 112, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1),
            # [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            MBConvBlock(112, 192, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),
            # [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
            MBConvBlock(192, 320, expand_ratio=6, kernel_size=3, stride=1),

            # bottleneck
            MBConvBlock(320, 320, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(320, 320, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(320, 320, expand_ratio=6, kernel_size=3, stride=1),

            # up
            DeconvBNReLU(320, 192),
            DeconvBNReLU(192, 112),
            DeconvBNReLU(112, 80),
            DeconvBNReLU(80, 40),
            DeconvBNReLU(40, 24),
            nn.ReflectionPad2d(1),
            nn.Conv2d(24, 3, 3, bias=True),
            nn.Tanh(),
        ]
        super(EfficientGenerator, self).__init__(*features)


def main():
    model = EfficientGenerator()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(y.size())
    print(numel(model))


def numel(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    main()

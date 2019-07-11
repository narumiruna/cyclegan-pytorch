import torch
from torch import nn


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *inputs):
        return inputs


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
        super(ConvBNReLU, self).__init__(*layers)


class ConvTranposeBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes):
        layers = [
            nn.ConvTranspose2d(in_planes, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
        super(ConvTranposeBNReLU, self).__init__(*layers)


class ResnetGenerator(nn.Sequential):

    def __init__(self):
        layers = [
            ConvBNReLU(3, 64, 7, 1),
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 256, 3, stride=2),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ConvTranposeBNReLU(256, 128),
            ConvTranposeBNReLU(128, 64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, bias=True),
            nn.Tanh(),
        ]
        super(ResnetGenerator, self).__init__(*layers)


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.features = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.features(x)


def numel(model):
    return sum(p.numel() for p in model.parameters())


def main():
    model = ResnetGenerator()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print(y.size())
    print(numel(model))

if __name__ == '__main__':
    main()

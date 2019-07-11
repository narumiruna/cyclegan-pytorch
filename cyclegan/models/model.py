from torch import nn


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *inputs):
        return inputs


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, stride=1, padding_mode='zeros', norm_type='batch'):
        super(ResidualBlock, self).__init__()

        # padding
        padding = (kernel_size - 1) // 2
        print(padding)
        padding_layer = Identity()
        if padding_mode == 'zeros':
            padding_layer = nn.ZeroPad2d(padding)
        elif padding_mode == 'reflect':
            padding_layer = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replicate':
            padding_layer = nn.ReplicationPad2d(padding)
        else:
            raise ValueError('padding_mode should be zeros, reflect or replicate')

        # norm
        # norm_layer
        bias = False
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(channels)
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(channels)
            bias = True
        else:
            raise ValueError('norm_type should be batch or instance')

        self.features = nn.Sequential(
            padding_layer,
            nn.Conv2d(channels, channels, kernel_size, stride=stride, bias=bias),
            norm_layer,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.features(x)


def main():
    import torch
    model = ResidualBlock(3, 5)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print(y.size())


if __name__ == '__main__':
    main()

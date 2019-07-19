from torch import nn


def numel(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

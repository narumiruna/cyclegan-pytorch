import sys

from ..utils import get_factory
from .discriminator import Discriminator
from .resnet import ResnetGenerator

ModelFactory = get_factory(sys.modules[__name__])

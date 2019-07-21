import sys

from ..utils import get_factory
from .image import ImageFolder, ImageFolderLoader, get_image_folder_loaders

DataFactory = get_factory(sys.modules[__name__])

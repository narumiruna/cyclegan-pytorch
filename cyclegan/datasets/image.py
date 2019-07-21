import os
from glob import glob

from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import pil_loader


class ImageFolder(data.Dataset):

    def __init__(self, root, extensions=None, transform=None):
        self.root = root
        self.extensions = extensions or ('.jpg', '.png')
        self.paths = self._glob_paths()
        self.transform = transform

    def _glob_paths(self):
        paths = []
        for ext in self.extensions:
            pattern = os.path.join(self.root, '*{}'.format(ext))
            paths.extend(glob(pattern))
        return paths

    def __getitem__(self, index):
        img = pil_loader(self.paths[index])

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.paths)


class ImageFolderLoader(data.DataLoader):

    def __init__(self, root, extensions=None, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = ImageFolder(root, extensions, transform)
        super(ImageFolderLoader, self).__init__(dataset, **kwargs)


def get_image_folder_loaders(root, batch_size, num_workers=0):
    loader_a = ImageFolderLoader(os.path.join(root, 'trainA'),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    loader_b = ImageFolderLoader(os.path.join(root, 'trainB'),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    return loader_a, loader_b


def main():
    loader_a = ImageFolderLoader('data/apple2orange/trainA', batch_size=32, shuffle=True, num_workers=8)
    loader_b = ImageFolderLoader('data/apple2orange/trainB', batch_size=32, shuffle=True, num_workers=8)
    for i, (x, y) in enumerate(zip(loader_a, loader_b)):
        print(i, x.size(), y.size())


if __name__ == '__main__':
    main()

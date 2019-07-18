import os
from glob import glob

from torch.utils import data
from torchvision.datasets.folder import pil_loader


class ImageFolder(data.Dataset):

    def __init__(self, root, extensions=('.jpg', '.png'), transform=None):
        self.root = root
        self.extensions = extensions
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


def main():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset_a = ImageFolder('data/apple2orange/trainA', transform=transform)
    trainset_b = ImageFolder('data/apple2orange/trainB', transform=transform)
    loader_a = data.DataLoader(trainset_a, batch_size=32, shuffle=True, num_workers=8)
    loader_b = data.DataLoader(trainset_b, batch_size=32, shuffle=True, num_workers=8)
    for i, (x, y) in enumerate(zip(loader_a, loader_b)):
        print(i, x.size(), y.size())


if __name__ == '__main__':
    main()

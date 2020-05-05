import glob
import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(batch_size=128, path_to_data='/home/johanm/Documents/data/mnist'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloader(batch_size=128,
                                  path_to_data='/home/johanm/Documents/data/fashion_mnist'):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,
                            path_to_data='/home/johanm/Documents/data/dsprites/dsprites.npz'):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
    return dsprites_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='/home/johanm/Documents/data/rendered_cropped_chairs_64/'):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize(64),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True, num_workers=8, pin_memory=True)
    return chairs_loader


def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='/home/johanm/Documents/data/chairs'):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
    return chairs_loader


def get_celeba_dataloader(batch_size=128, path_to_data='/home/johanm/Documents/data/celeba_64/celeba'):
    """CelebA dataloader with (64, 64) images."""
    dataset = datasets.ImageFolder(root=path_to_data,
                                transform=transforms.Compose([
                                   transforms.ToTensor(),
                               ]))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return train_loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0

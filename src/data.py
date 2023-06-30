import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Pixel statistics of all (train + test) CIFAR-10 images
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
AVG = (0.4914, 0.4822, 0.4465) # Mean
STD = (0.2023, 0.1994, 0.2010) # Standard deviation
CHW = (3, 32, 32) # Channel, height, width
CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset:torchvision.datasets,
        transform:torchvision.transforms.Compose|A.Compose|None=None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, label


def get_transform() -> dict[str, A.Compose]:
     return {
        'train': A.Compose([
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=AVG, mask_fill_value=None,
            ),
            A.Normalize(mean=AVG, std=STD),
            ToTensorV2(),
        ]),
        'test': A.Compose([
            A.Normalize(mean=AVG, std=STD),
            ToTensorV2(),
        ]),
    }


def get_dataset(
    transform:dict[str, torchvision.transforms],
) -> dict[str, torchvision.datasets]:
    return {
        'train': TransformedDataset(
            dataset=torchvision.datasets.CIFAR10('../data', train=True, download=True),
            transform=transform['train'],
        ),
        'test': TransformedDataset(
            dataset=torchvision.datasets.CIFAR10('../data', train=False, download=False),
            transform=transform['test'],
        ),
    }


def get_dataloader(
    dataset:torchvision.datasets,
    params:dict[str, bool|int],
) -> dict[str, torch.utils.data.DataLoader]:
    return {
        'train': torch.utils.data.DataLoader(dataset['train'], **params),
        'test': torch.utils.data.DataLoader(dataset['test'], **params),
    }

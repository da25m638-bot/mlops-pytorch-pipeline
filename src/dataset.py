import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset_name: str = "cifar10",
) -> tuple[DataLoader, DataLoader]:
 
    if dataset_name == "cifar10":
        cls = datasets.CIFAR10
    elif dataset_name == "fashion_mnist":
        cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = cls(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = cls(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader

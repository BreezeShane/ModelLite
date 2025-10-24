"""
    Data loader definition.
"""
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

from schemas.context_classes import DatasetContextConfig

__all__ = [
    "load_dataset"
]

def load_dataset(cfg: DatasetContextConfig):
    """ Load CIFAR10 dataset. """
    # Define data pre-processing.
    transform = transforms.Compose([
        transforms.Resize(cfg.transform.resize),
        transforms.ToTensor(),
        # ImageNet Normalization.
        transforms.Normalize(
            mean=cfg.transform.normalize.mean,
            std=cfg.transform.normalize.std)
    ])

    train_dataset = CIFAR10(root=cfg.root_dir, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=cfg.root_dir, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, _ = random_split(
        train_dataset, [cfg.subset_ratio, 1-cfg.subset_ratio], generator)
    test_dataset, _ = random_split(
        test_dataset, [cfg.subset_ratio, 1-cfg.subset_ratio], generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

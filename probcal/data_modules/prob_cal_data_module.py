from pathlib import Path
from typing import Sized, Union, Literal

import numpy as np
from lightning import LightningDataModule
from numpy import ndarray
from torch import nn
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, AutoAugment
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets import ImageDatasetWrapper


class ProbCalDataModule(LightningDataModule):
    full_dataset: Union[Dataset, Sized]
    train: Dataset
    val: Dataset
    test: Dataset

    shuffled_indices: ndarray
    train_indices: ndarray
    val_indices: ndarray
    test_indices: ndarray

    train_transforms: list[nn.Module] = []
    inference_transforms: list[nn.Module] = []

    IMG_SIZE = 224
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        full_dataset: Union[Sized, Dataset],
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        train_val_split: tuple[float, float] = (0.7, 0.1),
        seed=1998,
    ):
        super(ProbCalDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.full_dataset = full_dataset
        self._init_indices(train_val_split, seed)
        self._init_transforms()

    def _init_indices(self, train_val_split: tuple[float, float], seed=1998):
        num_instances = len(self.full_dataset)
        train_split, val_split = train_val_split
        num_train = int(train_split * num_instances)
        num_val = int(val_split * num_instances)
        generator = np.random.default_rng(seed=seed)
        self.shuffled_indices = generator.permutation(np.arange(num_instances))
        self.train_indices = self.shuffled_indices[:num_train]
        self.val_indices = self.shuffled_indices[num_train : num_train + num_val]
        self.test_indices = self.shuffled_indices[num_train + num_val :]

    def _init_transforms(self):
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        augment = AutoAugment()
        normalize = Normalize(mean=self.IMAGE_NET_MEAN, std=self.IMAGE_NET_STD)
        to_tensor = ToTensor()
        self.train_transforms = [resize, augment, to_tensor, normalize]
        self.inference_transforms = [resize, to_tensor, normalize]

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self.train = ImageDatasetWrapper(
            base_dataset=Subset(self.full_dataset, self.train_indices),
            transforms=Compose(self.train_transforms),
        )
        self.val = ImageDatasetWrapper(
            base_dataset=Subset(self.full_dataset, self.val_indices),
            transforms=Compose(self.inference_transforms),
        )
        self.test = ImageDatasetWrapper(
            base_dataset=Subset(self.full_dataset, self.test_indices),
            transforms=Compose(self.inference_transforms),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

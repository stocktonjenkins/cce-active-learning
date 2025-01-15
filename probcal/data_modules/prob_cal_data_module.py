import os
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
from probcal.enums import DatasetType
from probcal.custom_datasets import ImageDatasetWrapper, collate_fn

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
        dataset_type: DatasetType = DatasetType.IMAGE,
        train_val_split: tuple[float, float] = (0.7, 0.1),
        indices_save_dir: str | Path | None = None,
        seed=1998,
    ):
        super(ProbCalDataModule, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.full_dataset = full_dataset
        self.train_val_split = train_val_split
        self.indices_save_dir = (
            Path(indices_save_dir) if indices_save_dir is not None else None
        )
        self._init_indices(seed)
        self._init_transforms()

    def _init_indices(self, seed=1998):
        num_instances = len(self.full_dataset)
        train_split, val_split = self.train_val_split
        num_train = int(train_split * num_instances)
        num_val = int(val_split * num_instances)
        generator = np.random.default_rng(seed=seed)
        self.shuffled_indices = generator.permutation(np.arange(num_instances))
        self.train_indices = self.shuffled_indices[:num_train]
        self.val_indices = self.shuffled_indices[num_train : num_train + num_val]
        self.test_indices = self.shuffled_indices[num_train + num_val :]
        if self.indices_save_dir is not None:
            if not self.indices_save_dir.exists():
                os.makedirs(self.indices_save_dir)
            np.savez(
                file=self.indices_save_dir / "split_indices.npz",
                train=self.train_indices,
                val=self.val_indices,
                test=self.test_indices,
            )

    def _init_transforms(self):
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        augment = AutoAugment()
        normalize = Normalize(mean=self.IMAGE_NET_MEAN, std=self.IMAGE_NET_STD)
        to_tensor = ToTensor()
        self.train_transforms = [resize, augment, to_tensor, normalize]
        self.inference_transforms = [resize, to_tensor, normalize]

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if self.dataset_type == DatasetType.IMAGE:
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
        else:
            self.train = Subset(self.full_dataset, self.train_indices)
            self.val = Subset(self.full_dataset, self.val_indices)
            self.test = Subset(self.full_dataset, self.test_indices)

    def train_dataloader(self) -> DataLoader:
        if self.dataset_type == DatasetType.TEXT:
            return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            )

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.dataset_type == DatasetType.TEXT:
            return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            )
        
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.dataset_type == DatasetType.TEXT:
            return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            )
        
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

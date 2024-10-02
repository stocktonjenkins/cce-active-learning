from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets import FGNetDataset
from probcal.custom_datasets import ImageDatasetWrapper


class FGNetDataModule(L.LightningDataModule):

    IMG_SIZE = 224
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        ignore_grayscale: bool = True,
        surface_image_path: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.ignore_grayscale = ignore_grayscale
        self.surface_image_path = surface_image_path

    def prepare_data(self) -> None:
        # Force images to be downloaded.
        FGNetDataset(self.root_dir)

    def setup(self, stage: str):
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        augment = AutoAugment()
        normalize = Normalize(mean=self.IMAGE_NET_MEAN, std=self.IMAGE_NET_STD)
        to_tensor = ToTensor()
        train_transforms = Compose([resize, augment, to_tensor, normalize])
        inference_transforms = Compose([resize, to_tensor, normalize])

        full_dataset = FGNetDataset(
            root_dir=self.root_dir,
            ignore_grayscale=self.ignore_grayscale,
            surface_image_path=self.surface_image_path,
        )
        num_instances = len(full_dataset)
        generator = np.random.default_rng(seed=1998)
        shuffled_indices = generator.permutation(np.arange(num_instances))
        num_train = int(0.7 * num_instances)
        num_val = int(0.1 * num_instances)
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train : num_train + num_val]
        test_indices = shuffled_indices[num_train + num_val :]

        self.train = ImageDatasetWrapper(
            base_dataset=Subset(full_dataset, train_indices),
            transforms=train_transforms,
        )
        self.val = ImageDatasetWrapper(
            base_dataset=Subset(full_dataset, val_indices),
            transforms=inference_transforms,
        )
        self.test = ImageDatasetWrapper(
            base_dataset=Subset(full_dataset, test_indices),
            transforms=inference_transforms,
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

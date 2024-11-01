from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import GaussianBlur
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets import COCOPeopleDataset
from probcal.custom_datasets import ImageDatasetWrapper
from probcal.custom_datasets import LabelNoiseImageDatasetWrapper
from probcal.custom_datasets import MixupImageDatasetWrapper
from probcal.data_modules.prob_cal_data_module import ProbCalDataModule
from probcal.transforms import GaussianNoiseTransform
from probcal.transforms import MixUpTransform


class COCOPeopleDataModule(ProbCalDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(
            full_dataset=COCOPeopleDataset(
                root_dir,
                surface_image_path=surface_image_path,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def prepare_data(self) -> None:
        # Force images to be downloaded.
        COCOPeopleDataset(self.root_dir)

    @classmethod
    def denormalize(cls, tensor):
        # Clone the tensor so the original stays unmodified
        tensor = tensor.clone()

        # De-normalize by multiplying by the std and then adding the mean
        for t, m, s in zip(tensor, cls.IMAGE_NET_MEAN, cls.IMAGE_NET_STD):
            t.mul_(s).add_(m)

        return tensor


class OodBlurCocoPeopleDataModule(COCOPeopleDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(root_dir, batch_size, num_workers, persistent_workers, surface_image_path)

    def setup(self, stage, *args, **kwargs):
        if stage != "test":
            raise ValueError(f"Invalid stage: {stage}. Only 'test' is supported for OOD class")

        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        ood_transform = GaussianBlur(
            kernel_size=(5, 9), sigma=(kwargs["perturb"], kwargs["perturb"])
        )
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        inference_transforms = Compose([resize, ood_transform, to_tensor, normalize])
        num_instances = len(self.full_dataset)
        generator = np.random.default_rng(seed=1998)
        shuffled_indices = generator.permutation(np.arange(num_instances))
        num_train = int(0.7 * num_instances)
        num_val = int(0.1 * num_instances)
        test_indices = shuffled_indices[num_train + num_val :]

        self.test = ImageDatasetWrapper(
            base_dataset=Subset(self.full_dataset, test_indices),
            transforms=inference_transforms,
        )


class OodMixupCocoPeopleDataModule(COCOPeopleDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(root_dir, batch_size, num_workers, persistent_workers, surface_image_path)

    def setup(self, stage, *args, **kwargs):
        if stage != "test":
            raise ValueError(f"Invalid stage: {stage}. Only 'test' is supported for OOD class")

        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        inference_transforms = Compose([resize, to_tensor, normalize])
        mixup_transform = MixUpTransform(kwargs["perturb"])
        num_instances = len(self.full_dataset)
        generator = np.random.default_rng(seed=1998)
        shuffled_indices = generator.permutation(np.arange(num_instances))
        num_train = int(0.7 * num_instances)
        num_val = int(0.1 * num_instances)
        test_indices = shuffled_indices[num_train + num_val :]

        self.test = MixupImageDatasetWrapper(
            base_dataset=Subset(self.full_dataset, test_indices),
            transforms=inference_transforms,
            mixup_transform=mixup_transform,
        )


class OodLabelNoiseCocoPeopleDataModule(COCOPeopleDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(root_dir, batch_size, num_workers, persistent_workers, surface_image_path)

    def setup(self, stage, *args, **kwargs):
        if stage != "test":
            raise ValueError(f"Invalid stage: {stage}. Only 'test' is supported for OOD class")

        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        inference_transforms = Compose([resize, to_tensor, normalize])
        full_dataset = COCOPeopleDataset(
            self.root_dir,
            surface_image_path=self.surface_image_path,
        )
        num_instances = len(full_dataset)
        generator = np.random.default_rng(seed=1998)
        shuffled_indices = generator.permutation(np.arange(num_instances))
        num_train = int(0.7 * num_instances)
        num_val = int(0.1 * num_instances)
        test_indices = shuffled_indices[num_train + num_val :]

        noise_transform = GaussianNoiseTransform(**kwargs["perturb"])
        self.test = LabelNoiseImageDatasetWrapper(
            base_dataset=Subset(full_dataset, test_indices),
            transforms=inference_transforms,
            noise_transform=noise_transform,
        )

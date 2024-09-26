import torch
from torch.utils.data import Dataset


class ImageDatasetWrapper(Dataset):
    """A wrapper class that allows for assigning different transforms to different views of the same dataset (such as train/test)."""

    def __init__(self, base_dataset, transforms):
        super(ImageDatasetWrapper, self).__init__()
        self.base = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transforms(x), y


class MixupImageDatasetWrapper(ImageDatasetWrapper):
    def __init__(self, base_dataset, transforms, mixup_transform):
        super().__init__(base_dataset, transforms)
        self.mixup_transform = mixup_transform

    def __getitem__(self, idx):
        x1, y1 = self.base[idx]

        rand_idx = torch.randperm(len(self.base))[0].item()
        x2, y2 = self.base[rand_idx]

        x1 = self.transforms(x1)
        x2 = self.transforms(x2)

        x = self.mixup_transform(x1, x2)
        y = self.mixup_transform(y1, y2)

        return x, y


class LabelNoiseImageDatasetWrapper(ImageDatasetWrapper):
    def __init__(self, base_dataset, transforms, noise_transform):
        super().__init__(base_dataset, transforms)
        self.noise_transform = noise_transform

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = self.noise_transform(y)
        return self.transforms(x), y

import os
from pathlib import Path
from shutil import rmtree
from shutil import unpack_archive
from typing import Callable

import pandas as pd
import wget
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class FGNetDataset(Dataset):
    """FGNet images, labeled by the age of the individual pictured."""

    URL = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        ignore_grayscale: bool = True,
        surface_image_path: bool = False,
    ):
        """Create an instance of the FGNet dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            ignore_grayscale (bool, optional): Whether/not to ignore grayscale images in FGNet (there are 175 grayscale and 827 color) when serving data. Defaults to True.
            surface_image_path (bool, optional): Whether/not to return the image path along with the image and count in __getitem__.
        """
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "FGNET" / "images"
        self.ignore_grayscale = ignore_grayscale
        self.surface_image_path = surface_image_path

        if not self.root_dir.exists():
            os.makedirs(self.root_dir)
        if not self._already_downloaded():
            self._download()

        self.transform = transform
        self.target_transform = target_transform
        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return self.image_dir.exists() and any(self.image_dir.iterdir())

    def _download(self):
        if not self.image_dir.exists():
            print("Downloading data...")
            annotations_zip_fname = wget.download(self.URL, out=f"{self.root_dir}/fgnet.zip")
            unpack_archive(annotations_zip_fname, self.root_dir)
            rmtree(self.root_dir / "__MACOSX")
            rmtree(self.root_dir / "FGNET" / "Data_files")
            rmtree(self.root_dir / "FGNET" / "points")
            os.remove(self.root_dir / "fgnet.zip")

    def _get_instances_df(self) -> pd.DataFrame:
        instances = {"image_path": [], "age": []}
        for image_fname in os.listdir(self.image_dir):
            image_path = self.image_dir / image_fname
            if self.ignore_grayscale:
                if Image.open(image_path).mode != "RGB":
                    # There are 175 grayscale images that we choose to skip for now.
                    continue
            age = int(image_fname[4:6])
            instances["image_path"].append(str(self.image_dir / image_fname))
            instances["age"].append(age)
        return pd.DataFrame(instances)

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.instances.iloc[idx]
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        age = row["age"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            age = self.target_transform(age)
        if self.surface_image_path:
            return image, (image_path, age)
        else:
            return image, age

    def __len__(self):
        return len(self.instances)

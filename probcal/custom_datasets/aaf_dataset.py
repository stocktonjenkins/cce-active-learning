import csv
import os
from pathlib import Path
from shutil import move
from shutil import rmtree
from shutil import unpack_archive
from typing import Callable

import gdown
import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class AAFDataset(Dataset):
    """The All-Age-Faces (AAF) Dataset contains 13'322 face images (mostly Asian)
    distributed across all ages (from 2 to 80), including 7381 females and 5941 males."""

    DATA_URL = "https://drive.google.com/uc?id=1wa5qOHUZn9O3Zp1efVoTwkK7PWZu8FJS"

    def __init__(
        self,
        root_dir: str | Path,
        limit: int | None = None,
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        surface_image_path: bool = False,
    ):
        """Create an instance of the COCOPeople dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            limit (int | None, optional): Max number of images to download/use for this dataset. Defaults to None.
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            surface_image_path (bool, optional): Whether/not to return the image path along with the image and age in __getitem__.
        """

        self.root_dir = Path(root_dir)
        self.limit = limit
        self.image_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"
        self.annotations_csv_path = self.annotations_dir / "annotations.csv"
        self.surface_image_path = surface_image_path

        for dir in self.root_dir, self.image_dir, self.annotations_dir:
            if not dir.exists():
                os.makedirs(dir)

        if not self._already_downloaded():
            self._download()

        self.transform = transform
        self.target_transform = target_transform
        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return (
            self.annotations_dir.exists()
            and any(self.annotations_dir.iterdir())
            and self.image_dir.exists()
            and any(self.image_dir.iterdir())
        )

    def _download(self):

        # Download raw folder
        print("Downloading zipped file...")
        zip_file_name = "All-Age-Faces Dataset"
        output_path = str(self.root_dir) + "/" + zip_file_name + ".zip"
        gdown.download(self.DATA_URL, output_path, quiet=False, fuzzy=True)
        unpack_archive(output_path, self.root_dir)

        # Setup annotations files
        input_txt_file1 = str(self.root_dir / zip_file_name / "image sets" / "train.txt")
        input_txt_file2 = str(self.root_dir / zip_file_name / "image sets" / "val.txt")
        output_csv_file = str(self.annotations_csv_path)

        # Convert from txt to csv + transformations
        with open(input_txt_file1, "r") as txt_file1, open(
            input_txt_file2, "r"
        ) as txt_file2, open(output_csv_file, "w") as csv_file:

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["image_path", "age", "gender"])

            for line in txt_file1:
                filename, gender = line.strip().split()
                age = filename[-6:-4]
                csv_writer.writerow([filename, age, gender])

            for line in txt_file2:
                filename, gender = line.strip().split()
                age = filename[-6:-4]
                csv_writer.writerow([filename, age, gender])

        # Setup images folder
        src_folder = str(self.root_dir / zip_file_name / "original images")

        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(self.image_dir, filename)
            move(src_file, dest_file)

        # Clean up
        os.remove(output_path)
        rmtree(str(self.root_dir / zip_file_name))

    def _get_instances_df(self) -> pd.DataFrame:
        annotations = str(self.annotations_csv_path)
        return pd.read_csv(annotations)

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.instances.iloc[idx]
        image_path = str(self.image_dir / row["image_path"])
        image = Image.open(image_path)
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

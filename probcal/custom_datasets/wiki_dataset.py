import csv
import os
from pathlib import Path
from shutil import move, rmtree, unpack_archive
from typing import Callable

import gdown
import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import datetime as date
from dateutil.relativedelta import relativedelta

import tarfile


class WikiDataset(Dataset):
    """The Wiki Dataset contains face images with age and gender annotations."""

    DATA_URL = (
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"
    )

    def __init__(
        self,
        root_dir: str | Path,
        limit: int | None = None,
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        surface_image_path: bool = False,
    ):
        """Create an instance of the Wiki dataset.

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
        # Download tar file
        print("Downloading tar file...")
        tar_file_name = "wiki_crop.tar"
        output_path = str(self.root_dir / tar_file_name)
        gdown.download(self.DATA_URL, output_path, quiet=False, fuzzy=True)

        # Extract tar file
        print("Extracting tar file...")
        with tarfile.open(output_path, "r") as tar:
            tar.extractall(path=self.root_dir)

        # Load and process the .mat file
        wiki_mat = str(self.root_dir / "wiki_crop" / "wiki.mat")
        wiki_data = loadmat(wiki_mat)
        wiki = wiki_data["wiki"]

        wiki_photo_taken = wiki[0][0][1][0]
        wiki_full_path = wiki[0][0][2][0]
        wiki_gender = wiki[0][0][3][0]
        wiki_face_score1 = wiki[0][0][6][0]
        wiki_face_score2 = wiki[0][0][7][0]

        wiki_path = []
        for path in wiki_full_path:
            wiki_path.append(path[0])

        wiki_genders = []
        for n in range(len(wiki_gender)):
            if wiki_gender[n] == 1:
                wiki_genders.append("male")
            else:
                wiki_genders.append("female")

        wiki_dob = []
        for file in wiki_path:
            wiki_dob.append(file.split("_")[2])

        wiki_age = []
        for i in range(len(wiki_dob)):
            try:
                d1 = date.datetime.strptime(wiki_dob[i][0:10], "%Y-%m-%d")
                d2 = date.datetime.strptime(str(wiki_photo_taken[i]), "%Y")
                rdelta = relativedelta(d2, d1)
                diff = rdelta.years
            except Exception as ex:
                print(ex)
                diff = -1
            wiki_age.append(diff)

        final_wiki = np.vstack(
            (wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)
        ).T
        final_wiki_df = pd.DataFrame(final_wiki)
        final_wiki_df.columns = ["age", "gender", "path", "face_score1", "face_score2"]

        meta = final_wiki_df
        meta = meta[meta["face_score1"] != "-inf"]
        meta = meta[meta["face_score2"] == "nan"]
        meta = meta.drop(["face_score1", "face_score2"], axis=1)
        meta = meta.sample(frac=1)

        meta.to_csv(self.annotations_csv_path, index=False)

        # Setup images folder
        src_folder = str(self.root_dir / "wiki_crop")
        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(self.image_dir, filename)
            move(src_file, dest_file)

        # Clean up
        os.remove(output_path)
        rmtree(src_folder)

    def _get_instances_df(self) -> pd.DataFrame:
        annotations = str(self.annotations_csv_path)
        return pd.read_csv(annotations)

    def __getitem__(
        self, idx: int
    ) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.instances.iloc[idx]
        image_path = str(
            self.image_dir / row["path"]
        )  # Adjusted to remove 'wiki_crop/'
        image = Image.open(image_path)
        # Convert grayscale images to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

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

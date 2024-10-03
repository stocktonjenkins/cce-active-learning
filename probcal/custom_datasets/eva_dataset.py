import os
import subprocess
import zipfile
from glob import glob
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class EVADataset(Dataset):
    """
    EVA dataset with images voted on how asthetic they are (labeled with the average asthetic score for each image).\n
    """

    LABELS_CSV = "votes_filtered.csv"

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        surface_image_path: bool = False,
    ) -> None:
        """Create an instance of the EVA dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            surface_image_path (bool, optional): Whether/not to return the image path along with the image and count in __getitem__.
        """
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.surface_image_path = surface_image_path

        self.repo_url = "https://github.com/kang-gnak/eva-dataset"

        self.root_dir = Path(root_dir).joinpath("eva-dataset")
        self.labels_dir = self.root_dir.joinpath("data")
        self.image_dir = self.root_dir.joinpath("images", "EVA_together")

        if not self._check_for_eva_data():
            self._download_dataset()
            return

        if not self._check_for_eva_images():
            self._concatenate_and_extract_zip()

        # read the label data in from the data folder
        self.votes_filtered_df = pd.read_csv(self.labels_dir.joinpath(self.LABELS_CSV), sep="=")

        # find the average of all the votes to create the label for the image
        self.labels_df = (
            self.votes_filtered_df.groupby("image_id")["score"]
            .agg(["mean", "count", "std"])
            .reset_index()
        )
        self.labels_df.columns = ["image_id", "avg_score", "vote_count", "score_std"]

        # the file name for each image is just {image_id}.jpg
        self.labels_df["file_name"] = self.labels_df["image_id"].astype(str) + ".jpg"

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.labels_df.iloc[idx]
        image_path = self.image_dir.joinpath(row["file_name"])
        image = Image.open(image_path)
        image = self._ensure_rgb(image)
        score = row["avg_score"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            score = self.target_transform(score)
        if self.surface_image_path:
            return image, (image_path, score)
        else:
            return image, score

    def __len__(self):
        return len(self.labels_df)

    def _ensure_rgb(self, image: PILImage):
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _download_dataset(self):
        print(f"Downloading repository to {self.root_dir}")
        subprocess.run(["git", "clone", self.repo_url, self.root_dir], check=True)

    def _check_for_eva_data(self) -> bool:
        return (
            self.root_dir.exists()
            and self.labels_dir.exists()
            and self.labels_dir.joinpath(self.LABELS_CSV).exists()
            and self.root_dir.joinpath("images")
        )

    def _check_for_eva_images(self) -> bool:
        return self.image_dir.exists()

    def _concatenate_and_extract_zip(self):
        original_dir = os.getcwd()

        zip_prefix = "EVA_together.zip"

        # Change to the source directory
        os.chdir(self.root_dir.joinpath("images"))
        print(f"Changed working directory to: {os.getcwd()}")

        # Find all zip parts
        zip_parts = sorted(glob(f"{zip_prefix}.00*"))
        if not zip_parts:
            print(f"No zip parts found with prefix '{zip_prefix}' in the current directory.")
            return

        print(f"Found {len(zip_parts)} zip parts: {zip_parts}")

        # Concatenate zip parts
        full_zip = f"{zip_prefix}"
        with open(full_zip, "wb") as outfile:
            for zip_part in zip_parts:
                print(f"Concatenating: {zip_part}")
                with open(zip_part, "rb") as infile:
                    outfile.write(infile.read())

        print(f"Finished concatenating. Created: {full_zip}")

        # Extract the full zip file
        print(f"Extracting {full_zip}...")
        with zipfile.ZipFile(full_zip, "r") as zip_ref:
            zip_ref.extractall()

        print("Extraction complete.")

        if os.path.isfile(full_zip):
            os.remove(full_zip)
            print(f"Removed concatenated zip file: {full_zip}")

        os.chdir(original_dir)

    def _print_stats(self):
        """
        Print the statistics of the average scores for the dataset.
        """
        print("\nStatistics of average scores:")
        print(self.labels_df["avg_score"].describe())

    def _create_dist_graph(self):
        """
        Creates a graph of the distribution of average scores for the dataset.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.labels_df, x="avg_score", kde=True)
        plt.title("Distribution of Average Scores per Image")
        plt.xlabel("Average Score")
        plt.ylabel("Count")
        plt.show()

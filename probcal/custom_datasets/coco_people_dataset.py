import os
import re
from pathlib import Path
from shutil import unpack_archive
from typing import Callable

import pandas as pd
import wget

# from imgdl import download
from PIL import Image
from PIL.Image import Image as PILImage
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import httpx
from pathlib import Path
import asyncio


class COCOPeopleDataset(Dataset):
    """Subset of COCO with images containing people (labeled with the count of people in each image)."""

    ANNOTATIONS_URL = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    PERSON_CATEGORY_ID = 1

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
            surface_image_path (bool, optional): Whether/not to return the image path along with the image and count in __getitem__.
        """
        self.root_dir = Path(root_dir)
        self.limit = limit
        self.image_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"
        self.annotations_json_path = self.annotations_dir / "instances_train2017.json"
        self.surface_image_path = surface_image_path

        for dir in self.root_dir, self.image_dir, self.annotations_dir:
            if not dir.exists():
                os.makedirs(dir)

        if not self._already_downloaded():
            self._download()
        else:
            self.coco_api = COCO(self.annotations_json_path)
            self.image_ids = []
            self.image_paths = []
            for image_path in self.image_dir.iterdir():
                match = re.search(r"[1-9]\d*", image_path.stem)
                if match:
                    self.image_ids.append(int(match.group()))
                    self.image_paths.append(image_path)

        self.transform = transform
        self.target_transform = target_transform
        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return (
            self.annotations_json_path.exists()
            and self.image_dir.exists()
            and any(self.image_dir.iterdir())
        )

    async def download_image(url, path):
        """Download a single image asynchronously."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            with open(path, "wb") as file:
                file.write(response.content)

    async def download_images(image_urls, image_paths, n_workers=10):
        """Download multiple images concurrently."""
        tasks = [
            asyncio.create_task(download_image(url, path))
            for url, path in zip(image_urls, image_paths)
        ]

        # Use asyncio to limit concurrent downloads
        for i in range(0, len(tasks), n_workers):
            await asyncio.gather(*tasks[i : i + n_workers])

    def _download(self):
        if not self.annotations_json_path.exists():
            print("Downloading annotations file...")
            annotations_zip_fname = wget.download(
                self.ANNOTATIONS_URL, out=str(self.root_dir)
            )
            unpack_archive(annotations_zip_fname, self.root_dir)
        self.coco_api = COCO(self.annotations_json_path)

        self.image_ids = self.coco_api.getImgIds(catIds=self.PERSON_CATEGORY_ID)[
            : self.limit
        ]
        images = self.coco_api.loadImgs(self.image_ids)
        image_urls = []
        self.image_paths = []
        for image in images:
            image_urls.append(image["coco_url"])
            self.image_paths.append(str(self.image_dir / image["file_name"]))
        asyncio.run(download_images(image_urls, self.image_paths, n_workers=50))
        # self.image_paths = download(image_urls, paths=self.image_paths, n_workers=50)

    def _get_instances_df(self) -> pd.DataFrame:
        instances = {"image_path": [], "count": []}
        for image_id, image_path in zip(self.image_ids, self.image_paths):
            annotation_ids = self.coco_api.getAnnIds(image_id)
            annotations = self.coco_api.loadAnns(annotation_ids)
            num_people = len(
                [x for x in annotations if x["category_id"] == self.PERSON_CATEGORY_ID]
            )
            instances["image_path"].append(str(image_path))
            instances["count"].append(num_people)
        return pd.DataFrame(instances)

    def __getitem__(
        self, idx: int
    ) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.instances.iloc[idx]
        image_path = row["image_path"]
        image = Image.open(image_path)
        count = row["count"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            count = self.target_transform(count)
        if self.surface_image_path:
            return image, (image_path, count)
        else:
            return image, count

    def __len__(self):
        return len(self.instances)

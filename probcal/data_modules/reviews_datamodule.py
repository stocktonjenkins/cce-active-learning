import os
from typing import Iterable
from typing import Sized, Union, Literal

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from transformers import BatchEncoding
from transformers import DistilBertTokenizer

from probcal.custom_datasets import ReviewsDataset
from .prob_cal_data_module import ProbCalDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")


class ReviewsDataModule(ProbCalDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
    ):
        super().__init__(
            full_dataset = ReviewsDataset(
                root_dir,
            ),
            batch_size = batch_size,
            num_workers = num_workers,
            persistent_workers = persistent_workers,
        )

    def prepare_data(self) -> None:
        # Force check if reviews are already downloaded.
        ReviewsDataset(self.root_dir)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        print('new implementation')
        self.train = Subset(self.full_dataset, self.train_indices)
        self.val = Subset(self.full_dataset, self.val_indices)
        self.test = Subset(self.full_dataset, self.test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    @staticmethod
    def collate_fn(
        batch: Iterable[tuple[str, int]],
    ) -> tuple[BatchEncoding, torch.LongTensor]:
        all_text = []
        all_targets = []
        for x in batch:
            all_text.append(x[0])
            all_targets.append(x[1])

        input_tokens = tokenizer(
            all_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        targets_tensor = torch.tensor(all_targets)

        return input_tokens, targets_tensor

from pathlib import Path
from typing import Union, Sized

import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.procedures.base import IActiveLearningDataModuleDelegate
from probcal.custom_datasets import ImageDatasetWrapper
from probcal.data_modules.prob_cal_data_module import ProbCalDataModule
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class ActiveLearningDataModule(ProbCalDataModule):
    unlabeled: Dataset
    unlabeled_indices: ndarray
    config: ActiveLearningConfig

    def __init__(
        self,
        full_dataset: Union[Sized, Dataset],
        config: ActiveLearningConfig,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = False,
        train_val_split: tuple[float, float] = (0.7, 0.1),
        seed=1998,
    ):
        self.config = config  # must be initialized before super.__init__()
        self.train_val_split = train_val_split
        super(ActiveLearningDataModule, self).__init__(
            full_dataset=full_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            train_val_split=train_val_split,
            seed=seed,
        )

    def step(
        self, delegate: IActiveLearningDataModuleDelegate, model: DiscreteRegressionNN
    ):
        # The delegate will tell me how to get my next labeled set
        # i.e. random, CCEActiveLearning, etc.
        unlabeled_indices_to_label = delegate.get_next_label_set(
            unlabeled_indices=self.unlabeled_indices,
            k=self.config.label_k,
            model=model,
        )
        if self.config.update_validation_set:
            np.random.shuffle(unlabeled_indices_to_label)
            split_index = int(self.train_val_split[0] * len(unlabeled_indices_to_label))
            new_train_indices, new_val_indices = (
                unlabeled_indices_to_label[:split_index],
                unlabeled_indices_to_label[split_index:],
            )
            self.train_indices = np.union1d(self.train_indices, new_train_indices)
            self.val_indices = np.union1d(self.val_indices, new_val_indices)
        else:
            self.train_indices = np.union1d(
                self.train_indices, unlabeled_indices_to_label
            )
        self.unlabeled_indices = np.setdiff1d(
            self.unlabeled_indices, unlabeled_indices_to_label
        )

    def _init_indices(self, train_val_split: tuple[float, float], seed=1998):
        num_instances = len(self.full_dataset)
        train_split, val_split = train_val_split
        test_split = 1 - val_split - train_split
        generator = np.random.default_rng(seed=seed)
        self.shuffled_indices = generator.permutation(np.arange(num_instances))
        num_test = int(test_split * num_instances)
        self.test_indices, rest = (
            self.shuffled_indices[:num_test],
            self.shuffled_indices[num_test:],
        )
        labeled_indices = rest[: self.config.init_num_labeled]
        self.unlabeled_indices = rest[self.config.init_num_labeled:]
        num_train = int(train_split * self.config.init_num_labeled)
        self.train_indices, self.val_indices = (
            labeled_indices[:num_train],
            labeled_indices[num_train:],
        )
        self.unlabeled_indices = self.unlabeled_indices[:1000]

    def setup(self, stage):
        super().setup(stage)
        # Setup unlabeled... do we ignore the ys, (Rishabh): Ignore the y's, no need to remove them, we'll need them later anyways
        # or do we have to explicitly remove them?
        self.unlabeled = ImageDatasetWrapper(
            base_dataset=Subset(self.full_dataset, self.unlabeled_indices),
            transforms=Compose(self.inference_transforms),
        )

    def unlabeled_dataloader(self) -> DataLoader:
        return DataLoader(
            self.unlabeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

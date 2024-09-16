import random
from pathlib import Path
from typing import Type, Optional
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from probcal.data_modules import TabularDataModule, COCOPeopleDataModule
from probcal.enums import DatasetType, ImageDatasetName, TextDatasetName
from probcal.enums import HeadType
from probcal.models import GaussianNN
from probcal.models import NegBinomNN
from probcal.models import PoissonNN
from probcal.models.backbones import MLP
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.utils.configs import TrainingConfig

GLOBAL_DATA_DIR = 'data'


def get_model(config: TrainingConfig, return_initializer: bool = False) -> DiscreteRegressionNN:

    initializer: Type[DiscreteRegressionNN]

    if config.head_type == HeadType.GAUSSIAN:
        initializer = GaussianNN
    elif config.head_type == HeadType.POISSON:
        initializer = PoissonNN
    elif config.head_type == HeadType.NEGATIVE_BINOMIAL:
        initializer = NegBinomNN

    if config.dataset_type == DatasetType.TABULAR:
        backbone_type = MLP
        backbone_kwargs = {"input_dim": config.input_dim}
    backbone_kwargs["output_dim"] = config.hidden_dim

    model = initializer(
        backbone_type=backbone_type,
        backbone_kwargs=backbone_kwargs,
        optim_type=config.optim_type,
        optim_kwargs=config.optim_kwargs,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
    )
    if return_initializer:
        return model, initializer
    else:
        return model


def get_datamodule(
    dataset_type: DatasetType,
    dataset_spec: Path | ImageDatasetName,
    batch_size: int,
    num_workers: Optional[int] = 8
) -> L.LightningDataModule:
    if dataset_type == DatasetType.TABULAR:
        return TabularDataModule(
            dataset_path=dataset_spec,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )
    elif dataset_type == DatasetType.IMAGE:
        if dataset_spec == ImageDatasetName.MNIST:
            return ValueError("MNIST not supported.")
        elif dataset_spec == ImageDatasetName.COCO_PEOPLE:
            return COCOPeopleDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "coco_people"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
    elif dataset_type == DatasetType.TEXT:
        if dataset_spec == TextDatasetName.REVIEWS:
            return ValueError("Reviews not supported.")


def fix_random_seed(random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


def get_chkp_callbacks(chkp_dir: Path, chkp_freq: int) -> list[ModelCheckpoint]:
    temporal_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        every_n_epochs=chkp_freq,
        filename="{epoch}",
        save_top_k=-1,
        save_last=True,
    )
    best_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        monitor="val_loss",
        every_n_epochs=1,
        filename="best_loss",
        save_top_k=1,
        enable_version_counter=False,
    )
    best_mae_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        monitor="val_mae",
        every_n_epochs=1,
        filename="best_mae",
        save_top_k=1,
        enable_version_counter=False,
    )
    return [
        temporal_checkpoint_callback,
        best_loss_checkpoint_callback,
        best_mae_checkpoint_callback,
    ]

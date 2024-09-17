import random
from pathlib import Path
from typing import Type, Optional, Union
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from probcal.data_modules import TabularDataModule, COCOPeopleDataModule, OodCocoPeopleDataModule
from probcal.enums import DatasetType, ImageDatasetName, TextDatasetName
from probcal.enums import HeadType
from probcal.models import GaussianNN
from probcal.models import NegBinomNN
from probcal.models import PoissonNN
from probcal.models.backbones import MLP, ViT, MNISTCNN
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.utils.configs import TrainingConfig, TestConfig
from probcal.utils.generic_utils import partialclass

GLOBAL_DATA_DIR = 'data'


def get_model(config: Union[TrainingConfig, TestConfig], return_initializer: bool = False) -> DiscreteRegressionNN:

    initializer: Type[DiscreteRegressionNN]

    if config.head_type == HeadType.MEAN:
        initializer = MeanNN
    elif config.head_type == HeadType.GAUSSIAN:
        if hasattr(config, "beta_scheduler_type") and config.beta_scheduler_type is not None:
            initializer = partialclass(
                GaussianNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = GaussianNN
    elif config.head_type == HeadType.FAITHFUL_GAUSSIAN:
        #initializer = FaithfulGaussianNN
        raise NotImplementedError("Faithful Gaussian not implemented.")
    elif config.head_type == HeadType.NATURAL_GAUSSIAN:
        #initializer = NaturalGaussianNN
        raise NotImplementedError("Natural Gaussian not implemented.")
    elif config.head_type in (HeadType.POISSON, HeadType.POISSON_GLM):
        initializer = PoissonNN
    elif config.head_type in (HeadType.DOUBLE_POISSON, HeadType.DOUBLE_POISSON_GLM):
        raise NotImplementedError("Double Poisson not implemented.")
        """if config.beta_scheduler_type is not None:
            initializer = partialclass(
                DoublePoissonNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = DoublePoissonNN"""
    elif config.head_type in (HeadType.NEGATIVE_BINOMIAL, HeadType.NEGATIVE_BINOMIAL_GLM):
        initializer = NegBinomNN
    elif config.head_type == HeadType.MULTI_CLASS:
        if config.dataset_type == DatasetType.IMAGE:
            if config.dataset_spec != ImageDatasetName.MNIST:
                raise ValueError("Can only train MultiClassNN on MNIST or Reviews.")
            else:
                discrete_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif config.dataset_type == DatasetType.TEXT:
            # Must be Amazon Reviews.
            discrete_values = [1, 2, 3, 4, 5]
        else:
            raise ValueError("Can only train MultiClassNN on MNIST or Reviews.")

        initializer = partialclass(
            MultiClassNN,
            discrete_values=discrete_values,
        )

    if config.dataset_type == DatasetType.TABULAR:
        if config.head_type in (
            HeadType.POISSON_GLM,
            HeadType.NEGATIVE_BINOMIAL_GLM,
            HeadType.DOUBLE_POISSON_GLM,
        ):
            backbone_type = Identity
        elif "collision" in str(config.dataset_spec):
            backbone_type = LargerMLP
        else:
            backbone_type = MLP
        backbone_kwargs = {"input_dim": config.input_dim}
    elif config.dataset_type == DatasetType.TEXT:
        backbone_type = DistilBert
        backbone_kwargs = {}
    elif config.dataset_type == DatasetType.IMAGE:
        if config.dataset_path_or_spec == ImageDatasetName.MNIST:
            backbone_type = MNISTCNN
        elif config.dataset_path_or_spec == ImageDatasetName.COCO_PEOPLE:
            backbone_type = ViT
        else:
            backbone_type = MobileNetV3
        backbone_kwargs = {}
    backbone_kwargs["output_dim"] = config.hidden_dim

    if isinstance(config, TrainingConfig):
        model = initializer(
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=config.optim_type,
            optim_kwargs=config.optim_kwargs,
            lr_scheduler_type=config.lr_scheduler_type,
            lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        )
    elif isinstance(config, TestConfig):
        model = initializer(
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
        )
    else:
        raise ValueError("Config must be either TrainingConfig or TestConfig.")

    if return_initializer:
        return model, initializer
    else:
        return model


def get_datamodule(
    dataset_type: DatasetType,
    dataset_path_or_spec: Path | ImageDatasetName,
    batch_size: int,
    num_workers: Optional[int] = 8
) -> L.LightningDataModule:
    if dataset_type == DatasetType.TABULAR:
        return TabularDataModule(
            dataset_path=dataset_path_or_spec,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )
    elif dataset_type == DatasetType.IMAGE:
        if dataset_path_or_spec == ImageDatasetName.MNIST:
            return ValueError("MNIST not supported.")
        elif dataset_path_or_spec == ImageDatasetName.COCO_PEOPLE:
            return COCOPeopleDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "coco_people"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif dataset_path_or_spec == ImageDatasetName.OOD_COCO_PEOPLE:
            return OodCocoPeopleDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "coco_people"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
    elif dataset_type == DatasetType.TEXT:
        if dataset_path_or_spec == TextDatasetName.REVIEWS:
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

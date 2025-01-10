import os
import random
from pathlib import Path
from typing import Optional
from typing import Type

import lightning as L
import numpy as np
import torch
import yaml
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from probcal.active_learning.configs import CheckpointType
from probcal.data_modules import AAFDataModule
from probcal.data_modules import COCOPeopleDataModule
from probcal.data_modules import EVADataModule
from probcal.data_modules import FGNetDataModule
from probcal.data_modules import OodBlurCocoPeopleDataModule
from probcal.data_modules import OodLabelNoiseCocoPeopleDataModule
from probcal.data_modules import OodMixupCocoPeopleDataModule
from probcal.data_modules import ReviewsDataModule
from probcal.data_modules import TabularDataModule
from probcal.data_modules import WikiDataModule
from probcal.data_modules.prob_cal_data_module import ProbCalDataModule
from probcal.enums import DatasetType
from probcal.enums import HeadType
from probcal.enums import ImageDatasetName
from probcal.enums import TextDatasetName
from probcal.models import DoublePoissonNN
from probcal.models import FaithfulGaussianNN
from probcal.models import FeedForwardNN
from probcal.models import GaussianNN
from probcal.models import NaturalGaussianNN
from probcal.models import NegBinomNN
from probcal.models import PoissonNN
from probcal.models.backbones import Backbone
from probcal.models.backbones import DistilBert
from probcal.models.backbones import LargerMLP
from probcal.models.backbones import MLP
from probcal.models.backbones import MNISTCNN
from probcal.models.backbones import MobileNetV3
from probcal.models.backbones import ViT
from probcal.models.regression_nn import RegressionNN
from probcal.utils.configs import EvaluationConfig
from probcal.utils.configs import TrainingConfig
from probcal.utils.generic_utils import partialclass

GLOBAL_DATA_DIR = "data"


def get_backbone_from_config(
    config: TrainingConfig,
) -> tuple[type[Backbone], dict[str, float]]:
    backbone_type = config.backbone_type
    subclasses = Backbone.__subclasses__()
    Class = list(filter(lambda c: c.__name__ == backbone_type, subclasses))[0]
    if Class is None:
        raise ValueError(
            f"{backbone_type} is not a valid backbone. Must be one of [{subclasses}]"
        )
    kwargs = {}
    if config.dataset_type == DatasetType.TABULAR:
        kwargs.update({"input_dim": config.input_dim})
    return Class, kwargs


def get_backbone_from_dataset(
    config: TrainingConfig,
) -> tuple[type[Backbone], dict[str, float]]:
    backbone_kwargs = {}
    if config.dataset_type == DatasetType.TABULAR:
        if "collision" in str(config.dataset_path_or_spec):
            backbone_type = LargerMLP
        else:
            backbone_type = MLP
        backbone_kwargs.update({"input_dim": config.input_dim})
    elif config.dataset_type == DatasetType.TEXT:
        backbone_type = DistilBert
    elif config.dataset_type == DatasetType.IMAGE:
        if config.dataset_path_or_spec == ImageDatasetName.MNIST:
            backbone_type = MNISTCNN
        elif config.dataset_path_or_spec == ImageDatasetName.COCO_PEOPLE:
            backbone_type = ViT
        else:
            backbone_type = MobileNetV3
    else:
        raise ValueError(
            f"Unable to determine backbone from dataset_type: {config.dataset_type}"
        )
    return backbone_type, backbone_kwargs


def get_model(
    config: TrainingConfig | EvaluationConfig, return_initializer: bool = False
) -> RegressionNN | tuple[RegressionNN, type[RegressionNN]]:
    initializer: Type[RegressionNN]

    if config.head_type == HeadType.GAUSSIAN:
        if (
            hasattr(config, "beta_scheduler_type")
            and config.beta_scheduler_type is not None
        ):
            initializer = partialclass(
                GaussianNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = GaussianNN
    elif config.head_type == HeadType.FAITHFUL_GAUSSIAN:
        initializer = FaithfulGaussianNN
    elif config.head_type == HeadType.NATURAL_GAUSSIAN:
        initializer = NaturalGaussianNN
    elif config.head_type == HeadType.FEED_FORWARD:
        initializer = FeedForwardNN
    elif config.head_type == HeadType.POISSON:
        initializer = PoissonNN
    elif config.head_type == HeadType.DOUBLE_POISSON:
        initializer = DoublePoissonNN
    elif config.head_type == HeadType.NEGATIVE_BINOMIAL:
        initializer = NegBinomNN
    else:
        raise ValueError(f"Head type {config.head_type} not recognized.")

    if config.backbone_type:
        backbone_type, backbone_kwargs = get_backbone_from_config(config)
    else:
        backbone_type, backbone_kwargs = get_backbone_from_dataset(config)

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
    elif isinstance(config, EvaluationConfig):
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
    num_workers: Optional[int] = 8,
) -> L.LightningDataModule | ProbCalDataModule:
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
        elif dataset_path_or_spec == ImageDatasetName.OOD_BLUR_COCO_PEOPLE:
            return OodBlurCocoPeopleDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "coco_people"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif dataset_path_or_spec == ImageDatasetName.OOD_MIXUP_COCO_PEOPLE:
            return OodMixupCocoPeopleDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "coco_people"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif dataset_path_or_spec == ImageDatasetName.OOD_LABEL_NOISE_COCO_PEOPLE:
            return OodLabelNoiseCocoPeopleDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "coco_people")
            )
        elif dataset_path_or_spec == ImageDatasetName.AAF:
            return AAFDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "aaf"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif dataset_path_or_spec == ImageDatasetName.WIKI:
            return WikiDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "wiki"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif dataset_path_or_spec == ImageDatasetName.EVA:
            return EVADataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "eva"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif dataset_path_or_spec == ImageDatasetName.FG_NET:
            return FGNetDataModule(
                root_dir=GLOBAL_DATA_DIR,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
    elif dataset_type == DatasetType.TEXT:
        if dataset_path_or_spec == TextDatasetName.REVIEWS:
            return ReviewsDataModule(
                root_dir=os.path.join(GLOBAL_DATA_DIR, "reviews"),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )


def fix_random_seed(random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


def get_chkp_callbacks(chkp_dir: Path, chkp_freq: int) -> list[Callback]:
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
        filename=CheckpointType.BEST_LOSS.value,
        save_top_k=1,
        enable_version_counter=False,
    )
    best_mae_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        monitor="val_mae",
        every_n_epochs=1,
        filename=CheckpointType.BEST_MAE.value,
        save_top_k=1,
        enable_version_counter=False,
    )
    return [
        temporal_checkpoint_callback,
        best_loss_checkpoint_callback,
        best_mae_checkpoint_callback,
    ]


def from_yaml(fpath):
    with open(fpath, "r") as f:
        return yaml.safe_load(f)

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import TypeVar, Union, Any
from probcal.training.train_model import train_procedure
from probcal.utils.configs import TrainingConfig

import lightning as L

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
    IActiveLearningDataModuleDelegate,
    ModelAccuracyResults,
)
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.models.feed_forward_regression_nn import FFRegressionNN
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.utils.experiment_utils import get_model, get_datamodule, get_chkp_callbacks
from logging import Logger
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger


T = TypeVar("T")
EvalState = Union[ActiveLearningEvaluationResults, T]


def get_logger(
    train_config: TrainingConfig,
    logger_type: str,
    log_dirname: str,
    version: str,
) -> Logger | bool:
    logger_args = {
        "save_dir": train_config.log_dir,
        "name": log_dirname,
        "version": version,
    }
    return (
        CSVLogger(**logger_args)
        if logger_type == "csv"
        else TensorBoardLogger(**logger_args)
        if logger_type == "tboard"
        else True
    )


class DropoutProcedure(ActiveLearningProcedure[ActiveLearningProcedure]):
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: FFRegressionNN
    ) -> np.ndarray:
        """
        Randomly choose the next set of indices to add to label set
        Args:
            unlabeled_indices: np.ndarray
            k: int
            model: DiscreteRegressionNN

        Returns:
            A random subset of unlabeled indices.
        """
        print("training selection model")

        _train_config = TrainingConfig.from_yaml(
            "configs/train/aaf/feed_forward_cfg.yaml"
        )
        ff_model = get_model(_train_config)

        ckpt = _train_config.chkp_dir / "selection_model"
        logger = get_logger(_train_config, "csv", ckpt / "logger", "dummy_gaussian")

        trainer = L.Trainer(
            devices="auto",
            accelerator=_train_config.accelerator_type.value,
            min_epochs=_train_config.num_epochs,
            max_epochs=_train_config.num_epochs,
            log_every_n_steps=5,
            check_val_every_n_epoch=1,
            enable_model_summary=False,
            callbacks=get_chkp_callbacks(ckpt, chkp_freq=_train_config.num_epochs),
            logger=logger,
            precision=_train_config.precision,
        )
        trainer.fit(model=ff_model, datamodule=self.dataset)
        print("selection model training complete")

        # ff_model_trainer = train_procedure(
        #         ff_model,
        #         datamodule=self.dataset,
        #         config=_train_config,
        #         callbacks=get_chkp_callbacks(
        #             ckpt, chkp_freq=_train_config.num_epochs
        #         ),
        #         logger='csv',
        #         al_iter=0,
        #     )

        unlabeled_dataloader = self.dataset.unlabeled_dataloader()
        confidence = self.get_confidence_score_from_model(
            ff_model,
            data_loader=unlabeled_dataloader,
        )
        assert confidence.shape[0] == len(unlabeled_indices)
        _, sampling_indices = torch.topk(
            confidence, k=min(k, unlabeled_indices.shape[0])
        )

        return unlabeled_indices[sampling_indices]

    @staticmethod
    def get_confidence_score_from_model(
        model: FFRegressionNN, data_loader: DataLoader
    ) -> torch.Tensor:
        with torch.no_grad():
            conf = []
            for inputs, _ in tqdm(
                data_loader, desc="doing forward pass to compute confidence..."
            ):
                print(model.device)
                y_hat = model.sample(inputs.to(model.device), num_samples=10)
                var = torch.var(y_hat, dim=1, correction=1)
                conf.append(var)

            return torch.cat(conf, dim=0).float()

import os.path
import torch
import random
import numpy as np
import shutil
from argparse import Namespace, ArgumentParser
from logging import Logger

from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.active_learning_logger.active_learning_average_cce_logger import (
    ActiveLearningAverageCCELogger,
)
from probcal.active_learning.active_learning_logger.active_learning_model_accuracy_logger import (
    ActiveLearningModelAccuracyLogger,
)
from probcal.active_learning.procedures import get_active_learning_procedure
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.data_modules.active_learning_data_module import ActiveLearningDataModule
from probcal.data_modules.prob_cal_data_module import ProbCalDataModule
from probcal.training.train_model import train_procedure
from probcal.utils.configs import TrainingConfig
from probcal.utils.experiment_utils import get_model, get_datamodule, get_chkp_callbacks
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

def setup_seed_params(cfg):
    """Setup random seeds and other torch details
    """
    if cfg.disable_debug_apis:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    return None

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


def pipeline(
    train_config: TrainingConfig,
    active_learn: ActiveLearningProcedure,
    logger_type: str,
    log_dirname: str,
):
    for al_iter in range(active_learn.config.num_iter):
        al_iter_name = f"al_iter_{al_iter}"
        model = get_model(_train_config)
        chkp_dir = train_config.chkp_dir / log_dirname / al_iter_name
        trainer = train_procedure(
            model,
            datamodule=active_learn.dataset,
            config=train_config,
            callbacks=(
                get_chkp_callbacks(chkp_dir, chkp_freq=train_config.num_epochs)
                if al_iter % active_learn.config.model_ckpt_freq == 0
                else None
            ),
            logger=(
                get_logger(train_config, logger_type, log_dirname, al_iter_name)
                if al_iter % active_learn.config.model_ckpt_freq == 0
                else None
            ),
            validation_rate=active_learn.config.model_ckpt_freq,
        )
        active_learn.eval(trainer, model)
        active_learn.step(model)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train-config", type=str)
    parser.add_argument("--al-config", type=str)
    parser.add_argument("--logger", type=str, default="csv", help="csv|tboard")
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    _train_config = TrainingConfig.from_yaml(args.train_config)
    al_config = ActiveLearningConfig.from_yaml(args.al_config)
    setup_seed_params(al_config.deterministic_settings)
    Procedure: type[ActiveLearningProcedure] = get_active_learning_procedure(al_config)
    module = get_datamodule(
        _train_config.dataset_type,
        _train_config.dataset_path_or_spec,
        _train_config.batch_size,
    )
    if not isinstance(module, ProbCalDataModule):
        raise ValueError(f"Given module is not supported: {module}")
    active_learning_data_module_args = {
        "full_dataset": module.full_dataset,
        "batch_size": _train_config.batch_size,
        # "num_workers": _train_config.num_workers,
        "config": al_config,  # Assuming config is part of TrainingConfig
        # "persistent_workers": _train_config.persistent_workers,  # Add this field to TrainingConfig if not present
    }
    _active_learn = Procedure(
        dataset=ActiveLearningDataModule(**active_learning_data_module_args),
        config=al_config,
    )
    _log_dirname = f"{al_config.procedure_type}__{_train_config.experiment_name}"
    os.makedirs(os.path.join("logs", _log_dirname), exist_ok=True)

    _active_learn.attach(
        [
            ActiveLearningModelAccuracyLogger(
                path=os.path.join("logs", _log_dirname, f"al_model_acc.log")
            ),
            ActiveLearningAverageCCELogger(
                path=os.path.join("logs", _log_dirname, f"al_model_calibration.log")
            ),
        ]
    )
    shutil.copy(args.al_config, os.path.join("logs", _log_dirname, "al_config.yaml"))
    pipeline(
        train_config=_train_config,
        active_learn=_active_learn,
        logger_type=args.logger,
        log_dirname=_log_dirname,
    )
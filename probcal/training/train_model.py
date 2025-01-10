import math
from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
from lightning import LightningDataModule, Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from probcal.lib.logging import WandBLoggingCallback
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.utils.configs import TrainingConfig
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


def train_procedure(
    model: DiscreteRegressionNN,
    datamodule: LightningDataModule,
    config: TrainingConfig,
    callbacks: list[Callback] | None,
    logger: WandbLogger | Logger | bool,
    validation_rate: int = 1,
    devices: list[int] | str | int = "auto",
):
    trainer = L.Trainer(
        devices=devices,
        accelerator=config.accelerator_type.value,
        max_epochs=config.num_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=validation_rate,
        enable_model_summary=False,
        callbacks=callbacks,
        logger=logger,
        precision=config.precision,
    )
    trainer.fit(model=model, datamodule=datamodule)
    return trainer


def main(config: TrainingConfig):
    fix_random_seed(config.random_seed)
    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
    )
    for i in range(config.num_trials):
        model = get_model(config)
        chkp_dir = config.chkp_dir / config.experiment_name / f"version_{i}"
        chkp_callbacks = get_chkp_callbacks(chkp_dir, config.chkp_freq)
        if config.early_stopping:
            chkp_callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))
        if config.wandb:
            logger = WandbLogger(
                project="probcal",
                entity="gvpatil-uw",
                name=config.experiment_name,
                log_model=False,
            )
            chkp_callbacks.append(
                WandBLoggingCallback(
                    exp_name=f"{config.experiment_name}:trial-{i+1}", logger=logger
                )
            )
        else:
            logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)
        train_procedure(
            model=model,
            datamodule=datamodule,
            config=config,
            callbacks=chkp_callbacks,
            logger=logger,
            devices=config.devices,
        )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(TrainingConfig.from_yaml(args.config))

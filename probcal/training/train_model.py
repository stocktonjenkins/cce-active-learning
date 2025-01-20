from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
from lightning import Callback
from lightning import LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import Logger

from probcal.active_learning.configs import CheckpointType
from probcal.lib.logging import WandBLoggingCallback
from probcal.models.regression_nn import RegressionNN
from probcal.utils.configs import TrainingConfig
from probcal.utils.experiment_utils import fix_random_seed, get_chkp_callbacks
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


def train_procedure(
    model: RegressionNN,
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
        max_epochs=config.max_epochs,
        min_epochs=config.min_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=validation_rate,
        enable_model_summary=False,
        callbacks=callbacks,
        logger=logger,
        precision=config.precision,
        gradient_clip_algorithm=(
            config.gradient_clip_algorithm.value
            if config.gradient_clip_algorithm is not None
            else None
        ),
    )
    trainer.fit(model=model, datamodule=datamodule)
    return trainer


def main(config: TrainingConfig, chkp_type: CheckpointType):
    fix_random_seed(config.random_seed)
    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
    )
    chkp_dir = config.chkp_dir / config.experiment_name
    for i in range(config.num_trials):
        model = get_model(config)
        chkp_callbacks = list(
            filter(
                lambda cb: cb.filename == chkp_type.value,
                get_chkp_callbacks(chkp_dir, chkp_freq=1),
            )
        )
        if config.early_stopping:
            chkp_callbacks.append(
                EarlyStopping(
                    monitor="val_loss", mode="min", patience=5, min_delta=0.01
                )
            )
        if config.wandb:
            logger = WandbLogger(
                project="probcal",
                entity="gvpatil-uw",
                name=config.experiment_name,
                log_model=False,
            )
            logger.watch(model, log="gradients", log_freq=1)
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
    parser.add_argument("--chkp-type", type=CheckpointType)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(TrainingConfig.from_yaml(args.config), chkp_type=args.chkp_type)

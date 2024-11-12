import math
from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
from lightning import LightningDataModule, Callback
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers import CSVLogger

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
    logger: Logger | bool,
    validation_rate: int = 1,
):
    trainer = L.Trainer(
        devices=[1],
        strategy="auto",
        accelerator=config.accelerator_type.value,
        min_epochs=config.num_epochs,
        max_epochs=config.num_epochs,
        log_every_n_steps=2,
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
        logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)
        train_procedure(
            model=model,
            datamodule=datamodule,
            config=config,
            callbacks=chkp_callbacks,
            validation_rate=math.ceil(config.num_epochs / 200),
            logger=logger,
        )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(TrainingConfig.from_yaml(args.config))

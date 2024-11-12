from argparse import Namespace, ArgumentParser


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
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.training.train_model import train_procedure
from probcal.utils.configs import TrainingConfig
from probcal.utils.experiment_utils import get_model, get_datamodule, get_chkp_callbacks


def pipeline(
    model: DiscreteRegressionNN,
    train_config: TrainingConfig,
    active_learn: ActiveLearningProcedure,
    # logger: Logger | bool,
):
    for al_iter in range(active_learn.config.num_iter):
        chkp_dir = (
            train_config.chkp_dir
            / train_config.experiment_name
            / f"{active_learn.__class__.__name__}_al_iter_{al_iter}"
        )
        train_procedure(  # TODO: make sure the parameters for the model reference update
            model,
            datamodule=active_learn.dataset,
            config=train_config,
            # Checkpoints for model at the END of each AL iteration
            callbacks=get_chkp_callbacks(chkp_dir, chkp_freq=train_config.num_epochs),
            # Don't know if we need to log the model over every AL iteration?
            # Can we just evaluate the performance at the end of training?
            logger=False,
        )
        # notify (log) after step in base class?
        active_learn.eval(model)  # use self.dataset to eval model
        # Update dataset object
        active_learn.step()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train-config", type=str)
    parser.add_argument("--al-config", type=str)
    parser.add_argument("--logger", type=str, default="csv", help="csv|tboard")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _train_config = TrainingConfig.from_yaml(args.train_config)
    al_config = ActiveLearningConfig.from_yaml(args.al_config)
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
    _active_learn.attach(
        [
            ActiveLearningModelAccuracyLogger(),
            ActiveLearningAverageCCELogger(),
            # Add logging to save the model's results at the end of each AL iteration
            #  - End product: chart -> model accuracy (vs number of labels)
            #  - End product: chart -> average CCE (vs number of labels)
            #  - others???
        ]
    )
    pipeline(
        model=get_model(_train_config),
        train_config=_train_config,
        active_learn=_active_learn,
        # logger=(
        #     CSVLogger(save_dir=_train_config.log_dir, name=_train_config.experiment_name)
        #     if args.logger == "csv" else
        #     TensorBoardLogger(save_dir=_train_config.log_dir, name=_train_config.experiment_name)
        #     if args.logger == "tboard"
        #     else True
        # )
    )

from argparse import Namespace, ArgumentParser


from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.logging.active_learning_average_cce_logger import (
    ActiveLearningAverageCCE,
)
from probcal.active_learning.logging.active_learning_model_accuracy_logger import (
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
            / f"{active_learn.__name__}_al_iter_{al_iter}"
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
    _train_config = TrainingConfig.from_yaml(args.config)
    al_config = ActiveLearningConfig.from_yaml(args.config)
    Procedure: type[ActiveLearningProcedure] = get_active_learning_procedure(al_config)
    module = get_datamodule(**_train_config.__dict__)
    if not isinstance(module, ProbCalDataModule):
        raise ValueError(f"Given module is not supported: {module}")
    _active_learn = Procedure(
        dataset=ActiveLearningDataModule(
            full_dataset=module.full_dataset,
            **_train_config.__dict__,
        ),
        config=al_config,
    )
    _active_learn.attach(
        [
            ActiveLearningModelAccuracyLogger(),
            ActiveLearningAverageCCE(),
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

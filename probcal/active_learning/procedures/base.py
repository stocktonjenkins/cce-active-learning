import abc
from abc import ABC
from typing import TypeVar, Union, Any

import lightning

from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
    IActiveLearningDataModuleDelegate,
    ModelAccuracyResults,
)
from probcal.data_modules.active_learning_data_module import ActiveLearningDataModule
from probcal.evaluation import CalibrationEvaluator
from probcal.lib.observer import Subject
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


T = TypeVar("T")
EvalState = Union[ActiveLearningEvaluationResults, T]


class ActiveLearningProcedure(
    Subject[EvalState],
    IActiveLearningDataModuleDelegate,
    ABC,
):
    dataset: ActiveLearningDataModule
    config: ActiveLearningConfig
    _iteration: int = 0

    def __init__(self, dataset: ActiveLearningDataModule, config: ActiveLearningConfig):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.cal_evaluator = CalibrationEvaluator(config.settings)

    def eval(
        self,
        trainer: lightning.Trainer,
        model: DiscreteRegressionNN | None = None,
        best_path: str | None = None,
    ):
        """
        Evaluate the given model in this active learning procedure.
            - Model accuracy
            - Model calibration?
            - Other?
        Also updates the state to send to observers.
        Args:
            trainer: the lightning trainer used to train the model.
            model: the regression model to evaluate OR
            best_path: the path of the best model to evaluate
        Returns:

        """
        assert (
            model is not None or best_path is not None
        ), "`model` or `best_path` must be defined."
        results = trainer.test(
            model=model, ckpt_path=best_path, datamodule=self.dataset
        )
        model_accuracy_results = ModelAccuracyResults(**results[0])
        eval_dict = {
            "model_accuracy_results": model_accuracy_results,
            "iteration": self._iteration,
            "train_set_size": self.dataset.train_indices.shape[0],
            "val_set_size": self.dataset.val_indices.shape[0],
            "unlabeled_set_size": self.dataset.unlabeled_indices.shape[0],
        }
        if self.config.measure_calibration:
            calibration_results = self.cal_evaluator(model, data_module=self.dataset)
            eval_dict["calibration_results"] = calibration_results
        eval_cls, rest = self._eval_ext(trainer, model, best_path)
        evaluation: EvalState = eval_cls(**{**eval_dict, **rest})
        self.update_state(evaluation)

    def _eval_ext(
        self,
        trainer: lightning.Trainer,
        model: DiscreteRegressionNN | None = None,
        best_path: str | None = None,
    ) -> tuple[type[EvalState], dict[str, Any]]:
        """
        Adds evaluation stats and defines the type of the evaluation result.
        It must extend `ActiveLearningEvaluationResults`
        """
        return ActiveLearningEvaluationResults, {}

    def step(self, model: DiscreteRegressionNN):
        """
        Update `self.dataset` to include pool the unlabeled samples
        from AL into the training pool
        Returns:
            None
        """
        self.dataset.step(self, model)
        self._iteration += 1
        self.notify()

    def update_state(self, evaluation: EvalState):
        """
        Update the subject state. Prep for notifying observers
        Args:
            Evaluation results of the active learning procedure
        """
        self._state = evaluation

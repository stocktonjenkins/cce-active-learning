import abc
from abc import ABC
from typing import TypeVar, Union

from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
    IActiveLearningDataModuleDelegate,
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

    def __init__(self, dataset: ActiveLearningDataModule, config: ActiveLearningConfig):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.cal_evaluator = CalibrationEvaluator(config.settings)

    def eval(self, model: DiscreteRegressionNN):
        """
        Evaluate the given model in this active learning procedure.
            - Model accuracy
            - Model calibration
            - Other?
        Also updates the state to send to observers.
        Args:
            model: the regression model to evaluate
        Returns:

        """
        evaluation = self._eval_impl(model)
        self.update_state(evaluation)

    @abc.abstractmethod
    def _eval_impl(self, model: DiscreteRegressionNN) -> EvalState:
        pass

    def step(self):
        """
        Update `self.dataset` to include pool the unlabeled samples
        from AL into the training pool
        Returns:
            None
        """
        self.dataset.step(self)
        self.notify()


    def update_state(self, evaluation: EvalState):
        """
        Update the subject state. Prep for notifying observers
        Args:
            Evaluation results of the active learning procedure
        """
        self._state = evaluation

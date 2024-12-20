import abc
from dataclasses import dataclass

import numpy as np
from probcal.evaluation.calibration_evaluator import CalibrationResults
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class IActiveLearningDataModuleDelegate(abc.ABC):
    @abc.abstractmethod
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: DiscreteRegressionNN
    ) -> np.ndarray:
        pass


@dataclass
class ModelAccuracyResults:
    test_loss: float
    test_mae: float
    test_rmse: float
    nll: float


@dataclass
class ActiveLearningEvaluationResults:
    kth_trial: int
    iteration: int
    train_set_size: int
    val_set_size: int
    unlabeled_set_size: int
    model_accuracy_results: ModelAccuracyResults
    calibration_results: CalibrationResults | None = None

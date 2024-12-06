import abc
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import List
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
    iteration: int
    train_set_size: int
    val_set_size: int
    unlabeled_set_size: int
    calibration_results: CalibrationResults
    model_accuracy_results: ModelAccuracyResults


@dataclass
class RandomProcedureResults(ActiveLearningEvaluationResults):
    pass

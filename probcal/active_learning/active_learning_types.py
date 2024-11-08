import abc
from dataclasses import dataclass

import numpy as np

from probcal.evaluation.calibration_evaluator import CalibrationResults


class IActiveLearningDataModuleDelegate(abc.ABC):
    @abc.abstractmethod
    def get_next_label_set(self, unlabeled_indices: np.ndarray, k: int) -> np.ndarray:
        pass



@dataclass
class ModelAccuracyResults:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class ActiveLearningEvaluationResults:
    calibration_results: CalibrationResults
    model_accuracy_results: ModelAccuracyResults


@dataclass
class RandomProcedureResults(ActiveLearningEvaluationResults):
    random: int

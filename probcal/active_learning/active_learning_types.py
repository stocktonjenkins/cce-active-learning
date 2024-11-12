import abc
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import List
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

    @classmethod
    def from_predictions(cls, y_true: List[int], y_pred: List[int]) -> "ModelAccuracyResults":
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return cls(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
        )


@dataclass
class ActiveLearningEvaluationResults:
    calibration_results: CalibrationResults
    model_accuracy_results: ModelAccuracyResults


@dataclass
class RandomProcedureResults(ActiveLearningEvaluationResults):
    pass


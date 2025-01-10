import numpy as np

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.models.regression_nn import RegressionNN


class RandomProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: RegressionNN
    ) -> np.ndarray:
        """
        Randomly choose the next set of indices to add to label set
        Args:
            unlabeled_indices: np.ndarray
            k: int
            model: RegressionNN

        Returns:
            A random subset of unlabeled indices.
        """
        return np.random.choice(
            unlabeled_indices, size=min(k, unlabeled_indices.shape[0]), replace=False
        )

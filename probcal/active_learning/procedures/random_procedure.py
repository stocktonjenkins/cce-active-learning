import numpy as np

from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.active_learning.active_learning_types import (
    ModelAccuracyResults,
    RandomProcedureResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class RandomProcedure(ActiveLearningProcedure[RandomProcedureResults]):
    def get_next_label_set(self, unlabeled_indices: np.ndarray, k: int) -> np.ndarray:
        """
        Randomly choose the next set of indices to add to label set
        Args:
            unlabeled_indices: np.ndarray
            k: int

        Returns:
            A random subset of unlabeled indices.
        """
        return np.random.choice(unlabeled_indices, size=k, replace=False)

    def _eval_impl(self, model: DiscreteRegressionNN) -> RandomProcedureResults:
        calibration_results = self.cal_evaluator(model, self.dataset)

        model_accuracy_results = None
        print(f"Model accuracy results: {model_accuracy_results}")
        return RandomProcedureResults(
            calibration_results=calibration_results,
            model_accuracy_results=model_accuracy_results,
            random=42,  # Example random value, replace with actual logic if needed
        )
    
    def step(self):
        """
        Update `self.dataset` to include pool the unlabeled samples
        from AL into the training pool
        Returns:
            None
        """
        self.dataset.step(self)
        self.notify()
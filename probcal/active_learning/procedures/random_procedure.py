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
        # TODO: evaluate model and return results. Use self.dataset.test_dataloader()
        # results = self.cal_evaluator(model, data_module=self.dataset)
        # model_accuracy_results = ModelAccuracyResults.from_predictions(y_true, y_pred)
        breakpoint()
        return RandomProcedureResults(
            calibration_results=None,
            model_accuracy_results=None,
        )

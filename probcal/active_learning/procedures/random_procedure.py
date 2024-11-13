import lightning
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
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: DiscreteRegressionNN
    ) -> np.ndarray:
        """
        Randomly choose the next set of indices to add to label set
        Args:
            unlabeled_indices: np.ndarray
            k: int
            model: DiscreteRegressionNN

        Returns:
            A random subset of unlabeled indices.
        """
        return np.random.choice(unlabeled_indices, size=k, replace=False)

    def _eval_impl(
        self, trainer: lightning.Trainer, model: DiscreteRegressionNN
    ) -> RandomProcedureResults:
        calibration_results = self.cal_evaluator(model, self.dataset)
        results = trainer.test(model, datamodule=self.dataset)
        model_accuracy_results = ModelAccuracyResults(**results[0])
        return RandomProcedureResults(
            iteration=self._iteration,
            train_set_size=self.dataset.train_indices.shape[0],
            val_set_size=self.dataset.val_indices.shape[0],
            unlabeled_set_size=self.dataset.unlabeled_indices.shape[0],
            calibration_results=calibration_results,
            model_accuracy_results=model_accuracy_results,
        )

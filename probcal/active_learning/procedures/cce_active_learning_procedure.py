import numpy as np
import torch
from torch.utils.data import DataLoader

from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.active_learning.types import (
    ModelAccuracyResults,
    CCEProcedureResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class CCEProcedure(ActiveLearningProcedure[CCEProcedureResults]):
    def get_next_label_set(self, model: DiscreteRegressionNN, unlabeled_indices: np.ndarray, k: int) -> np.ndarray:
        """
        Randomly choose the next set of indices to add to label set
        Args:
            unlabeled_indices: np.ndarray
            k: int

        Returns:
            A random subset of unlabeled indices.
        """
        train_dataloader = DataLoader(self.dataset[self.dataset.train_indices], batch_size=self.dataset.batch_size, shuffle=False)
        unlabeled_dataloader = DataLoader(self.dataset[self.dataset.unlabeled_indices], batch_size=self.dataset.batch_size, shuffle=False)
        cce_unlabeled = self.cal_evaluator.compute_mcmd_unlabeled(model, unlabeled_data_loader=unlabeled_dataloader, data_loader=train_dataloader)
        assert cce_unlabeled.shape[0] == len(unlabeled_indices)
        _, sampling_indices = torch.topk(cce_unlabeled, k)

        return sampling_indices

    def _eval_impl(self, model: DiscreteRegressionNN) -> CCEProcedureResults:
        results = self.cal_evaluator(model, data_module=self.dataset)
        return CCEProcedureResults(
            calibration_results=results,
            model_accuracy_results=ModelAccuracyResults(),
            random=42,
        )

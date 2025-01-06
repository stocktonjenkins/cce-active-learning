import numpy as np
import torch

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class WeightedACEProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    """AL procedure where we select the next train batch using CCE values as the sampling weights."""

    def get_next_label_set(
        self,
        unlabeled_indices: np.ndarray,
        k: int,
        model: DiscreteRegressionNN,
    ) -> np.ndarray:
        model = model.to("cpu")
        train_dataloader = self.dataset.train_dataloader()
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()
        cce_unlabeled = self.cal_evaluator.compute_mcmd_unlabeled(
            model,
            unlabeled_data_loader=unlabeled_dataloader,
            data_loader=train_dataloader,
        )
        n = len(unlabeled_indices)
        assert cce_unlabeled.shape[0] == n

        sampling_probs = cce_unlabeled / cce_unlabeled.sum()
        num_samples = min(k, n)
        sampling_indices = torch.multinomial(sampling_probs, num_samples).detach().cpu().numpy()

        return unlabeled_indices[sampling_indices]

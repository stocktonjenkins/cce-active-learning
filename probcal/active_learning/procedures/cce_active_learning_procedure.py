import numpy as np
import torch

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class CCEProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
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
        assert cce_unlabeled.shape[0] == len(unlabeled_indices)
        _, sampling_indices = torch.topk(
            cce_unlabeled, k=min(k, unlabeled_indices.shape[0])
        )

        return unlabeled_indices[sampling_indices]

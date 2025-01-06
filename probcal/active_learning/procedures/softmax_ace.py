import numpy as np
import torch

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.data_modules.active_learning_data_module import ActiveLearningDataModule
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.training.hyperparam_schedulers import CosineAnnealingScheduler


class SoftmaxACEProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    """AL procedure where we select the next train batch using CCE values as the sampling logits (passed through a tempered softmax)."""

    def __init__(
        self,
        dataset: ActiveLearningDataModule,
        config: ActiveLearningConfig,
        tau_0: float = 0.1,
        tau_1: float = 0.001,
    ):
        super().__init__(dataset=dataset, config=config)
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.tau_scheduler = CosineAnnealingScheduler(tau_0, tau_1, config.num_al_iter)

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

        tau = self.tau_scheduler.current_value
        sampling_probs = torch.softmax(cce_unlabeled / tau, dim=-1)
        num_samples = min(k, n)
        sampling_indices = (
            torch.multinomial(sampling_probs, num_samples).detach().cpu().numpy()
        )
        self.tau_scheduler.step()

        return unlabeled_indices[sampling_indices]

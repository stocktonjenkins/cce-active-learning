import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.models.regression_nn import RegressionNN


class ConfidenceProcedure(ActiveLearningProcedure[ActiveLearningProcedure]):
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
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()
        confidence = self.get_confidence_score_from_model(
            model,
            data_loader=unlabeled_dataloader,
        )
        assert confidence.shape[0] == len(unlabeled_indices)
        _, sampling_indices = torch.topk(
            confidence, k=min(k, unlabeled_indices.shape[0])
        )

        return unlabeled_indices[sampling_indices]

    @staticmethod
    def get_confidence_score_from_model(
        model: RegressionNN, data_loader: DataLoader
    ) -> torch.Tensor:
        with torch.no_grad():
            conf = []
            for inputs, _ in tqdm(
                data_loader, desc="doing forward pass to compute confidence..."
            ):
                y_hat = model.predict(inputs.to(model.device))
                (mu, var) = torch.split(y_hat, [1, 1], dim=-1)
                conf.append(var.flatten())

            return torch.cat(conf).float()

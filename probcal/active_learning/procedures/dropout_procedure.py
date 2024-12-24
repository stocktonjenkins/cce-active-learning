import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.models.feed_forward_regression_nn import FFRegressionNN


class DropoutProcedure(ActiveLearningProcedure[ActiveLearningProcedure]):
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: FFRegressionNN
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
        model: FFRegressionNN, data_loader: DataLoader
    ) -> torch.Tensor:
        with torch.no_grad():
            conf = []
            for inputs, _ in tqdm(
                data_loader, desc="doing forward pass to compute confidence..."
            ):
                print(model.device)
                y_hat = model.sample(inputs.to(model.device), num_samples=10)
                var = torch.var(y_hat, dim=1, correction=1)
                conf.append(var)

            return torch.cat(conf, dim=0).float()
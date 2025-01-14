import numpy as np
from sklearn.metrics import pairwise_distances

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.models.regression_nn import RegressionNN


class CoreSetProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)

    def get_next_label_set(
        self,
        unlabeled_indices: np.ndarray,
        k: int,
        model: RegressionNN,
    ) -> np.ndarray:
        """
        Choose the next set of indices to add to the label set based on Fisher Information.

        Args:
            model: RegressionNN
            unlabeled_indices: np.ndarray
            k: int

        Returns:
            A subset of unlabeled indices selected based on Fisher Information.
        """
        train_dataloader = self.dataset.train_dataloader()
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()

        # get low-rank point-wise fishers
        xt_unlabeled = self.get_embedding(model, unlabeled_dataloader).numpy()
        xt_labeled = self.get_embedding(model, train_dataloader).numpy()

        chosen = self.furthest_first(xt_unlabeled, xt_labeled, k)
        return unlabeled_indices[chosen]

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

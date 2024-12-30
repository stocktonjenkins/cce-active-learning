import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import gc
from tqdm import tqdm
from copy import copy as copy
from copy import deepcopy as deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from sklearn.metrics import pairwise_distances


class BatchBALDProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def __init__(self, dataset, config, noise_sigma: float = 0.0):
        super().__init__(dataset, config)
        self.noise_sigma = noise_sigma
        self.l = None

    def get_next_label_set(
        self,
        unlabeled_indices: np.ndarray,
        k: int,
        model: DiscreteRegressionNN,
    ) -> np.ndarray:
        """
        Choose the next set of indices to add to the label set based on MaxDet.

        Args:
            model: DiscreteRegressionNN
            unlabeled_indices: np.ndarray
            k: int

        Returns:
            A subset of unlabeled indices selected based on MaxDet.
        """
        train_dataloader = self.dataset.train_dataloader()
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()

        # get embeddings
        xt_unlabeled = self.get_embedding(model, unlabeled_dataloader).numpy()
        xt_labeled = self.get_embedding(model, train_dataloader).numpy()

        chosen = self.max_det_selection(xt_unlabeled, xt_labeled, k)
        return unlabeled_indices[chosen]

    def max_det_selection(self, X, X_set, n):
        """
        Implement the MaxDet selection method to select points that maximize the determinant.

        Args:
            X: np.ndarray
            X_set: np.ndarray
            n: int

        Returns:
            Indices of the selected points.
        """
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            diag = np.tile(float("inf"), m)
        else:
            kernel_matrix = pairwise_distances(X, X_set, metric="euclidean")
            diag = np.diag(kernel_matrix) + self.noise_sigma**2

        idxs = []

        for i in range(n):
            idx = diag.argmax()
            idxs.append(idx)
            l = None if self.l is None else self.l[:, : len(idxs)]
            lTl = 0.0 if l is None else l @ l[idx, :]
            mat_col = kernel_matrix[idx, :]
            if self.noise_sigma > 0.0:
                mat_col[idx] += self.noise_sigma**2
            update = (1.0 / np.sqrt(diag[idx])) * (mat_col - lTl)
            diag -= update**2
            if self.l is None:
                self.l = update[:, None]
            else:
                self.l = np.hstack([self.l, update[:, None]])
            diag[idx] = -np.inf  # ensure that the index is not selected again

        return idxs

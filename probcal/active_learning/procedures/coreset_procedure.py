import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gc
from tqdm import tqdm
from copy import copy as copy
from copy import deepcopy as deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from sklearn.metrics import pairwise_distances


class CoreSetProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)

    def get_embedding(self, model, dataloader):
        loader_te = dataloader
        embedding = []
        model.eval()
        with torch.no_grad():
            print("Getting embedings")
            for inputs, _ in tqdm(loader_te):
                emb = model.get_last_layer_representation(inputs)
                embedding.append(emb.data.cpu())
        embedding = torch.cat(embedding, dim=0)

        return embedding

    def get_next_label_set(
        self,
        unlabeled_indices: np.ndarray,
        k: int,
        model: DiscreteRegressionNN,
    ) -> np.ndarray:
        """
        Choose the next set of indices to add to the label set based on Fisher Information.

        Args:
            model: DiscreteRegressionNN
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
import lightning
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
    ModelAccuracyResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class BadgeProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def __init__(self, dataset, config, subset_size=None):
        super().__init__(dataset, config)
        self.subset_size = subset_size

    def get_next_label_set(
        self,
        unlabeled_indices: np.ndarray,
        k: int,
        model: DiscreteRegressionNN,
    ) -> np.ndarray:
        """
        Choose the next set of indices to add to the label set based on Badge sampling.
        
        Args:
            model: DiscreteRegressionNN
            unlabeled_indices: np.ndarray
            k: int

        Returns:
            A subset of unlabeled indices selected based on Badge sampling.
        """
        model = model.to("cpu")
        
        # Get the DataLoader for unlabeled data
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()
        
        # Compute gradient embeddings for unlabeled data
        grad_embedding = model.get_grad_representations(unlabeled_dataloader)
        
        # Select top-k samples using k-means++
        chosen_indices = kmeans_plusplus(grad_embedding.numpy(), k, rng=np.random.default_rng())
        
        return unlabeled_indices[chosen_indices]

    def _eval_impl(self, trainer: lightning.Trainer, model: DiscreteRegressionNN) -> ActiveLearningEvaluationResults:
        """
        Evaluate the model and return the results, including calibration results and model accuracy results.
        
        Args:
            trainer: pl.Trainer
            model: DiscreteRegressionNN

        Returns:
            ActiveLearningEvaluationResults
        """
        # Evaluate the model using the calibration evaluator
        calibration_results = self.cal_evaluator(model, data_module=self.dataset)
        
        # Use the trainer to test the model
        results = trainer.test(model, datamodule=self.dataset)
        model_accuracy_results = ModelAccuracyResults(**results[0])
        
        return ActiveLearningEvaluationResults(
            calibration_results=calibration_results,
            model_accuracy_results=model_accuracy_results,
            iteration=self._iteration,
            train_set_size=self.dataset.train_indices.shape[0],
            val_set_size=self.dataset.val_indices.shape[0],
            unlabeled_set_size=self.dataset.unlabeled_indices.shape[0],
        )

    def step(self, model: DiscreteRegressionNN):
        """
        Update `self.dataset` to include pool the unlabeled samples
        from AL into the training pool.
        
        Returns:
            None
        """
        self.dataset.step(self, model)
        self._iteration += 1
        self.notify()


def kmeans_plusplus(X, n_clusters, rng):
    """
    K-means++ initialization for selecting top-k samples.
    
    Args:
        X: np.ndarray, gradient embeddings.
        n_clusters: int, number of clusters.
        rng: np.random.Generator, random number generator.

    Returns:
        indices: list of selected indices.
    """
    # Start with highest grad norm since it is the "most uncertain"
    grad_norm = np.linalg.norm(X, ord=2, axis=1)
    idx = np.argmax(grad_norm)

    all_distances = pairwise_distances(X, X)

    indices = [idx]
    centers = [X[idx]]
    dist_mat = []
    for _ in range(1, n_clusters):
        # Compute the distance of the last center to all samples
        dist = all_distances[indices[-1]]

        dist_mat.append(dist)
        # Get the distance of each sample to its closest center
        min_dist = np.min(dist_mat, axis=0)
        min_dist_squared = min_dist**2
        if np.all(min_dist_squared == 0):
            raise ValueError('All distances to the centers are zero!')
        # sample idx with probability proportional to the squared distance
        p = min_dist_squared / np.sum(min_dist_squared)
        if np.any(p[indices] != 0):
            print('Already sampled centers have probability', p)
        idx = rng.choice(range(len(X)), p=p.squeeze())
        indices.append(idx)
        centers.append(X[idx])
    return indices
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class BAITProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def __init__(
        self,
        dataset,
        config,
        fisher_approximation="full",
        fisher_batch_size=32,
        device="cpu",
    ):
        super().__init__(dataset, config)
        self.fisher_approximation = fisher_approximation
        self.fisher_batch_size = fisher_batch_size
        self.device = device

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
        # Set the model to evaluation mode
        model.eval()

        # Get the DataLoader for unlabeled data
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()

        # Compute Fisher Information for unlabeled data
        fisher_unlabeled, repr_unlabeled = self.compute_fisher_information(
            model, unlabeled_dataloader
        )
        # print(repr_unlabeled.shape)
        # Compute Fisher Information for labeled data
        labeled_dataloader = self.dataset.train_dataloader()
        fisher_labeled, repr_labeled = self.compute_fisher_information(
            model, labeled_dataloader
        )
        # print(repr_labeled.shape)
        # Compute fisher_labeled using the provided code snippet
        fisher_labeled = self.compute_fisher_labeled(repr_labeled)

        # Select top-k samples based on Fisher Information
        num_labeled = len(self.dataset.train_indices)
        lmb = 1.0  # Regularization parameter, adjust as needed
        chosen_indices = select_topk(
            repr_unlabeled,
            min(k, unlabeled_indices.shape[0]),
            fisher_unlabeled,
            fisher_labeled,
            lmb,
            num_labeled,
        )

        return unlabeled_indices[chosen_indices]

    @staticmethod
    def compute_fisher_information(
        model: DiscreteRegressionNN, dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Fisher Information for the data.

        Args:
            model: DiscreteRegressionNN
            dataloader: DataLoader

        Returns:
            A tensor of Fisher Information scores and representations for the data.
        """
        fisher_information = []
        representations = []

        for data, _ in dataloader:
            data.requires_grad = True  # Ensure the input tensor requires gradients

            # Forward pass through the backbone to get the last layer representation
            with torch.no_grad():
                x = model.backbone(data)

            # Compute the Fisher Information score
            fisher_info = torch.sum(x**2, dim=1)
            fisher_information.append(fisher_info)
            representations.append(x)

        # Clear cache to free up memory
        torch.cuda.empty_cache()

        return torch.cat(fisher_information), torch.cat(representations)

    def compute_fisher_labeled(self, repr_labeled: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fisher Information for the labeled data using the provided code snippet.

        Args:
            repr_labeled: Tensor of representations for the labeled data.

        Returns:
            A tensor of Fisher Information for the labeled data.
        """
        fisher_dim = (repr_labeled.size(-1), repr_labeled.size(-1))
        fisher_labeled = torch.zeros(fisher_dim).to(self.device)
        dl = DataLoader(
            TensorDataset(repr_labeled),
            batch_size=self.fisher_batch_size,
            shuffle=False,
        )
        print(len(dl))
        i = 0
        for batch in dl:
            i += 1
            if i % 10 == 0:
                print(i)
            repr_batch = batch[0].to(self.device)
            # print(repr_batch.shape)
            if self.fisher_approximation == "full":
                repr_batch = repr_batch.view(
                    repr_batch.size(0), -1, repr_batch.size(-1)
                )
                # print(repr_batch.shape)
                term = torch.matmul(repr_batch.transpose(1, 2), repr_batch)
                fisher_labeled += torch.mean(term, dim=0)
            elif self.fisher_approximation == "block_diag":
                repr_batch = repr_batch.view(-1, 10, 10, fisher_dim[0])
                term = torch.einsum("nkhd,mkhe->hde", repr_batch, repr_batch)
                fisher_labeled += term / len(repr_batch)
            elif self.fisher_approximation == "diag":
                term = torch.mean(torch.sum(repr_batch**2, dim=1), dim=0)
                fisher_labeled += term
            else:
                raise NotImplementedError()

        return fisher_labeled


def select_topk(repr_unlabeled, acq_size, fisher_all, fisher_labeled, lmb, num_labeled):
    device = fisher_all.device

    # Efficient computation of the objective (trace rotation & Woodbury identity)
    repr_unlabeled = repr_unlabeled.to(device)
    if repr_unlabeled.dim() == 2:
        repr_unlabeled = repr_unlabeled.unsqueeze(1)  # Add a rank dimension

    dim = repr_unlabeled.size(-1)
    rank = repr_unlabeled.size(-2)

    fisher_labeled = fisher_labeled * num_labeled / (num_labeled + acq_size)
    M_0 = lmb * torch.eye(dim, device=device) + fisher_labeled
    M_0_inv = torch.inverse(M_0)
    print(M_0_inv.shape)
    # repr_unlabeled = repr_unlabeled * np.sqrt(acq_size / (num_labeled + acq_size))
    A = torch.inverse(
        torch.eye(rank, device=device)
        + repr_unlabeled @ M_0_inv @ repr_unlabeled.transpose(1, 2)
    )
    tmp = (
        repr_unlabeled
        @ M_0_inv
        @ fisher_all
        @ M_0_inv
        @ repr_unlabeled.transpose(1, 2)
        @ A
    )
    scores = torch.diagonal(tmp, dim1=-2, dim2=-1).sum(-1)
    chosen = scores.topk(acq_size).indices
    return chosen

from typing import Type

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Metric

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.custom_torchmetrics import AverageNLL
from probcal.models.backbones import Backbone
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.training.losses import natural_gaussian_nll


class NaturalGaussianNN(DiscreteRegressionNN):
    """A neural network that learns the natural parameters of a Gaussian distribution over each regression target (conditioned on the input).

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
    """

    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a NaturalGaussianNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(NaturalGaussianNN, self).__init__(
            loss_fn=natural_gaussian_nll,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Linear(self.backbone.output_dim, 2)
        self.nll = AverageNLL()
        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (eta_1, eta_2), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        raw = self.head(h)
        y_hat = torch.stack([raw[:, 0], -0.5 * F.softplus(raw[:, 1])], dim=-1)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, var), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()
        return y_hat

    def _sample_impl(
        self, y_hat: torch.Tensor, training: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        """Sample from this network's posterior predictive distributions for a batch of data (as specified by y_hat).

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.
            num_samples (int, optional): Number of samples to take from each posterior predictive. Defaults to 1.

        Returns:
            torch.Tensor: Batched sample tensor, with shape (N, num_samples).
        """
        dist = self.posterior_predictive(y_hat, training)
        sample = dist.sample((num_samples,)).view(num_samples, -1).T
        return sample

    def _posterior_predictive_impl(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.Normal:
        eta_1, eta_2 = torch.split(y_hat, [1, 1], dim=-1)
        mu = self._natural_to_mu(eta_1, eta_2)
        var = self._natural_to_var(eta_2)
        mu = mu.flatten()
        var = var.flatten()
        std = torch.sqrt(var)
        dist = torch.distributions.Normal(loc=mu, scale=std)
        return dist

    def _point_prediction_impl(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        eta_1, eta_2 = torch.split(y_hat, [1, 1], dim=-1)
        mu = self._natural_to_mu(eta_1, eta_2)
        return mu.round()

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {"nll": self.nll}

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        targets = y.flatten()
        dist = self.posterior_predictive(y_hat)

        # We compute "probability" with the continuity correction (probability of +- 0.5 of the value).
        target_probs = dist.cdf(targets + 0.5) - dist.cdf(targets - 0.5)
        self.nll.update(target_probs)

    def _natural_to_mu(self, eta_1: torch.Tensor, eta_2: torch.Tensor) -> torch.Tensor:
        return -0.5 * (eta_1 / eta_2)

    def _natural_to_var(self, eta_2: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.reciprocal(eta_2)

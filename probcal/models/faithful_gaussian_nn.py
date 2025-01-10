from typing import Type

import torch
from torch import nn
from torchmetrics import Metric

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.custom_torchmetrics import AverageNLL
from probcal.models.backbones import Backbone
from probcal.models.regression_nn import RegressionNN
from probcal.training.losses import faithful_gaussian_nll


class FaithfulGaussianNN(RegressionNN):
    """Implementation of https://arxiv.org/abs/2212.09184.

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
        optim_type: OptimizerType | None = None,
        optim_kwargs: dict | None = None,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a FaithfulGaussianNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(FaithfulGaussianNN, self).__init__(
            loss_fn=faithful_gaussian_nll,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.mu_head = nn.Linear(self.backbone.output_dim, 1)
        self.logvar_head = nn.Linear(self.backbone.output_dim, 1)

        self.nll = AverageNLL()

        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, logvar), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h: torch.Tensor = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h.detach())
        y_hat = torch.cat((mu, logvar), dim=-1)
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

        # Apply torch.exp to the logvar dimension.
        output_shape = y_hat.shape
        reshaped = y_hat.view(-1, 2)
        y_hat = torch.stack([reshaped[:, 0], torch.exp(reshaped[:, 1])], dim=1).view(*output_shape)

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
        if training:
            mu, logvar = torch.split(y_hat, [1, 1], dim=-1)
            var = logvar.exp()
        else:
            mu, var = torch.split(y_hat, [1, 1], dim=-1)

        dist = torch.distributions.Normal(loc=mu.squeeze(), scale=var.sqrt().squeeze())
        return dist

    def _point_prediction_impl(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        mu, _ = torch.split(y_hat, [1, 1], dim=-1)
        return mu

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {"nll": self.nll}

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        mu, var = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        var = var.flatten()
        std = torch.sqrt(var)
        targets = y.flatten()

        # We compute "probability" with the continuity correction (probability of +- 0.5 of the value).
        dist = torch.distributions.Normal(loc=mu, scale=std)
        target_probs = dist.cdf(targets + 0.5) - dist.cdf(targets - 0.5)
        self.nll.update(target_probs)

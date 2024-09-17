from typing import Optional
from typing import Type

import torch
from scipy.stats import norm
from torch import nn
from torchmetrics import Metric

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.custom_torchmetrics import AverageNLL
from probcal.evaluation.custom_torchmetrics import RegressionECE
from probcal.models.backbones import Backbone
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.training.losses import gaussian_nll


class GaussianNN(DiscreteRegressionNN):
    """A neural network that learns the parameters of a Gaussian distribution over each regression target (conditioned on the input).

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
        optim_type: Optional[OptimizerType] = None,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_type: Optional[LRSchedulerType] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        """Instantiate a GaussianNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
            beta_scheduler_type (BetaSchedulerType | None, optional): If specified, the type of beta scheduler to use for training loss (if applicable). Defaults to None.
            beta_scheduler_kwargs (dict | None, optional): If specified, key-value argument specifications for the chosen beta scheduler, e.g. {"beta_0": 1.0, "beta_1": 0.5}. Defaults to None.
        """
        super(GaussianNN, self).__init__(
            loss_fn=gaussian_nll,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Linear(self.backbone.output_dim, 2)
        self.ece = RegressionECE(
            param_list=["loc", "scale"],
            rv_class_type=norm,
        )
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
        h = self.backbone(x)
        y_hat = self.head(h)  # Interpreted as (mu, logvar)
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

    def _point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        mu, _ = torch.split(y_hat, [1, 1], dim=-1)
        return mu.round()

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "regression_ece": self.ece,
            "nll": self.nll,
        }

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        mu, var = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        var = var.flatten()
        std = torch.sqrt(var)
        targets = y.flatten()

        self.ece.update({"loc": mu, "scale": std}, targets)

        # We compute "probability" with the continuity correction (probability of +- 0.5 of the value).
        dist = torch.distributions.Normal(loc=mu, scale=std)
        target_probs = dist.cdf(targets + 0.5) - dist.cdf(targets - 0.5)
        self.nll.update(target_probs)

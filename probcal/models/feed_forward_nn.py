from functools import partial
from typing import Optional
from typing import Type

import torch
from torch import nn
from torchmetrics import Metric

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.models.backbones import Backbone
from probcal.models.feed_forward_regression_nn import FFRegressionNN
from probcal.training.losses import mse_loss


class FeedForwardNN(FFRegressionNN):
    """A neural network for a regression target."""

    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: Optional[OptimizerType] = None,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_type: Optional[LRSchedulerType] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        """Instantiate a FeedForwardNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(FeedForwardNN, self).__init__(
            loss_fn=partial(
                mse_loss,
            ),
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Linear(self.backbone.output_dim, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).

        """
        h = self.dropout(self.backbone(x))
        y_hat = self.head(h)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).

        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()

        # Apply torch.exp to the logvar dimension.
        # output_shape = y_hat.shape
        # reshaped = y_hat.view(-1, 2)
        # y_hat = torch.stack([reshaped[:, 0], torch.exp(reshaped[:, 1])], dim=1).view(
        #     *output_shape
        # )

        return y_hat

    def _sample_impl(
        self, input: torch.Tensor, training: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        """Sample from this network's posterior predictive distributions for a batch of data (as specified by y_hat).

        Args:
            input (torch.Tensor): input on which you want to sample.
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.
            num_samples (int, optional): Number of samples to take from each posterior predictive. Defaults to 1.

        Returns:
            torch.Tensor: Batched sample tensor, with shape (N, num_samples).
        """
        output = []
        for i in range(num_samples):
            o = self._forward_impl(input)
            output.append(o)
        output = torch.cat(output, dim=1)
        return output

    # def _posterior_predictive_impl(
    #     self, y_hat: torch.Tensor, training: bool = False
    # ) -> torch.distributions.Normal:
    #     if training:
    #         mu, logvar = torch.split(y_hat, [1, 1], dim=-1)
    #         var = logvar.exp()
    #     else:
    #         mu, var = torch.split(y_hat, [1, 1], dim=-1)

    #     dist = torch.distributions.Normal(loc=mu.squeeze(), scale=var.sqrt().squeeze())
    #     return dist

    def _point_prediction_impl(
        self, y_hat: torch.Tensor, training: bool
    ) -> torch.Tensor:
        # mu, _ = torch.split(y_hat, [1, 1], dim=-1)
        return y_hat

    def get_last_layer_representation(self, x):
        """
        Get the representation at the last hidden layer before the output layer.

        Args:
            x: Input tensor.

        Returns:
            Tensor of the last layer's representations.
        """
        if hasattr(self, "layer1"):  # Check if `layer1` exists
            x = torch.relu(self.layer1(x))
            return x
        elif hasattr(self, "backbone"):  # Fallback for models with a backbone
            return self.backbone(x)
        else:
            raise AttributeError("Model does not have a 'layer1' or 'backbone'.")

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {"mse": self.mse}

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        self.mse = (y - y_hat) ** 2
        # mu, var = torch.split(y_hat, [1, 1], dim=-1)
        # mu = mu.flatten()
        # var = var.flatten()
        # std = torch.sqrt(var)
        # targets = y.flatten()

        # # We compute "probability" with the continuity correction (probability of +- 0.5 of the value).
        # dist = torch.distributions.Normal(loc=mu, scale=std)
        # target_probs = dist.cdf(targets + 0.5) - dist.cdf(targets - 0.5)
        # self.nll.update(target_probs)

    def on_train_epoch_end(self):
        if self.beta_scheduler is not None:
            self.beta_scheduler.step()
            self.loss_fn = partial(mse_loss, beta=self.beta_scheduler.current_value)
        super().on_train_epoch_end()

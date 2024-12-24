from typing import Callable
from typing import Optional
from typing import Type
import wandb
import lightning as L
import torch
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric
from torch.utils.data import DataLoader

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.models.backbones import Backbone
from probcal.random_variables.discrete_random_variable import DiscreteRandomVariable


class FFRegressionNN(L.LightningModule):
    """Base class for discrete regression neural networks. Should not actually be used for prediction (needs to define `training_step` and whatnot).

    Attributes:
        backbone (Backbone): The backbone to use for feature extraction (before applying the regression head).
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: Optional[OptimizerType] = None,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_type: Optional[LRSchedulerType] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        project_name: str = "discrete-regression",  # WandB project name
        experiment_name: Optional[str] = None,
        log_model: bool = True,
    ):
        """Instantiate a regression NN.

        Args:
            loss_fn (Callable): The loss function to use for training this NN.
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(FFRegressionNN, self).__init__()

        self.backbone = backbone_type(**backbone_kwargs)
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_fn = loss_fn
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def configure_optimizers(self) -> dict:
        if self.optim_type is None:
            raise ValueError("Must specify an optimizer type.")
        if self.optim_type == OptimizerType.ADAM:
            optim_class = torch.optim.Adam
        elif self.optim_type == OptimizerType.ADAM_W:
            optim_class = torch.optim.AdamW
        elif self.optim_type == OptimizerType.SGD:
            optim_class = torch.optim.SGD
        optimizer = optim_class(self.parameters(), **self.optim_kwargs)
        optim_dict = {"optimizer": optimizer}

        if self.lr_scheduler_type is not None:
            if self.lr_scheduler_type == LRSchedulerType.COSINE_ANNEALING:
                lr_scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
            lr_scheduler = lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
            optim_dict["lr_scheduler"] = lr_scheduler

        return optim_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Batched output tensor, with shape (N, D_out)
        """
        return self._forward_impl(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        This method will often differ from `forward` in cases where
        the output used for training is in log (or some other modified)
        space for numerical convenience.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Batched output tensor, with shape (N, D_out)
        """
        return self._predict_impl(x)

    def sample(
        self, input: torch.Tensor, training: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        """Sample from this network's posterior predictive distributions for a batch of data .

        Args:
            input (torch.Tensor): input tensor to the model, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.
            num_samples (int, optional): Number of samples to take from each posterior predictive. Defaults to 1.

        Returns:
            torch.Tensor: Batched sample tensor, with shape (N, num_samples).
        """
        return self._sample_impl(input, training, num_samples)

    def posterior_predictive(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.Distribution | DiscreteRandomVariable:
        """Transform the network's outputs into the implied posterior predictive distribution.

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.

        Returns:
            torch.distributions.Distribution | DiscreteRandomVariable: The posterior predictive distribution.
        """
        return self._posterior_predictive_impl(y_hat, training)

    def point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        """Transform the network's output into a single discrete point prediction.

        This method will vary depending on the type of regression head (probabilistic vs. deterministic).
        For example, a gaussian regressor will return the `mean` portion of its output as its point prediction, rounded to the nearest integer.

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example.

        Returns:
            torch.Tensor: Point predictions for the true regression target, with shape (N, 1).
        """
        return self._point_prediction_impl(y_hat, training)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        with torch.no_grad():
            point_predictions = self.point_prediction(y_hat, training=True).flatten()
            self.train_rmse.update(point_predictions, y.flatten().float())
            self.train_mae.update(point_predictions, y.flatten().float())
            self.log("train_rmse", self.train_rmse, on_epoch=True)
            self.log("train_mae", self.train_mae, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        # Since we used the model's forward method, we specify training=True to get the proper transforms.
        point_predictions = self.point_prediction(y_hat, training=True).flatten()
        self.val_rmse.update(point_predictions, y.flatten().float())
        self.val_mae.update(point_predictions, y.flatten().float())
        self.log("val_rmse", self.val_rmse, on_epoch=True)
        self.log("val_mae", self.val_mae, on_epoch=True)
        return loss

    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y_hat = self.predict(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        point_predictions = self.point_prediction(y_hat, training=False).flatten()
        self.test_rmse.update(point_predictions, y.flatten().float())
        self.test_mae.update(point_predictions, y.flatten().float())
        self._update_addl_test_metrics_batch(x, y_hat, y.view(-1, 1).float())

        self.log("test_rmse", self.test_rmse, on_epoch=True)
        self.log("test_mae", self.test_mae, on_epoch=True)
        # for name, metric_tracker in self._addl_test_metrics_dict().items():
        #     self.log(name, metric_tracker, on_epoch=True)
        return loss

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self.predict(x)
        return y_hat

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _sample_impl(
        self, y_hat: torch.Tensor, training: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _posterior_predictive_impl(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.Distribution | DiscreteRandomVariable:
        raise NotImplementedError("Should be implemented by subclass.")

    def _point_prediction_impl(
        self, y_hat: torch.Tensor, training: bool
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        """Return a dict with the metric trackers used by this model beyond the default rmse/mae."""
        raise NotImplementedError("Should be implemented by subclass.")

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        """Update additional test metric states (beyond default rmse/mae) given a batch of inputs/outputs/targets.

        Args:
            x (torch.Tensor): Model inputs.
            y_hat (torch.Tensor): Model predictions.
            y (torch.Tensor): Model regression targets.
        """
        raise NotImplementedError("Should be implemented by subclass.")

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

    def get_grad_representations(
        self, dataloader: DataLoader, device: torch.device = "cpu"
    ) -> torch.Tensor:
        """
        Compute gradient embeddings for all samples in a DataLoader.

        Args:
            dataloader: DataLoader providing the data.
            device: Device for computation (e.g., "cpu" or "cuda").

        Returns:
            torch.Tensor: Gradient embeddings for all samples.
        """
        grad_embeddings = []

        # Ensure the model is in evaluation mode
        self.eval()
        self.to(device)
        print(len(dataloader))
        i = 0
        for batch in dataloader:
            i += 1
            print(i)
            inputs, _ = batch  # Unlabeled samples
            inputs = inputs.to(device)
            # Process each sample independently
            for x in inputs:
                x = x.unsqueeze(0)  # Add batch dimension for single sample

                # Forward pass
                outputs = self(x)
                _, logvar = torch.split(outputs, [1, 1], dim=-1)
                # Compute the loss with respect to the predicted label
                loss = logvar.mean()
                # Compute gradients
                self.zero_grad()
                loss.backward()  # Compute gradients for this sample

                # Extract gradients of the head layer
                grad = self.head.weight.grad  # Access gradients of the head layer
                grad_embeddings.append(grad.flatten().detach().cpu())

        return torch.stack(grad_embeddings)

from dataclasses import dataclass
from functools import partial
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import TypeAlias

import lightning as L
import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt
from open_clip import CLIP
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from probcal.enums import DatasetType
from probcal.evaluation.kernels import laplacian_kernel
from probcal.evaluation.kernels import polynomial_kernel
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.evaluation.metrics import compute_regression_ece
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


@dataclass
class CalibrationResults:
    input_grid_2d: np.ndarray
    regression_targets: np.ndarray
    mcmd_vals: np.ndarray
    mean_mcmd: float
    ece: float


KernelFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class CalibrationEvaluatorSettings:
    dataset_type: DatasetType = DatasetType.IMAGE
    mcmd_input_kernel: Literal["polynomial"] | KernelFunction = "polynomial"
    mcmd_output_kernel: Literal["rbf", "laplacian"] | KernelFunction = "rbf"
    mcmd_lmbda: float = 0.1
    mcmd_num_samples: int = 5
    ece_bins: int = 50
    ece_weights: Literal["uniform", "frequency"] = "frequency"
    ece_alpha: float = 1.0


class CalibrationEvaluator:
    """Helper object to evaluate the calibration of a neural net."""

    def __init__(self, settings: CalibrationEvaluatorSettings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._image_preprocess = None
        self._tokenizer = None

    @torch.inference_mode()
    def __call__(
        self, model: DiscreteRegressionNN, data_module: L.LightningDataModule
    ) -> CalibrationResults:
        data_module.prepare_data()
        data_module.setup("test")
        test_dataloader = data_module.test_dataloader()

        print("Computing MCMD...")
        mcmd_vals, grid, targets = self.compute_mcmd(
            model, test_dataloader, return_grid=True, return_targets=True
        )

        print("Running TSNE to project grid to 2d...")
        grid_2d = TSNE().fit_transform(grid.detach().cpu().numpy())

        print("Computing ECE...")
        ece = self.compute_ece(model, test_dataloader)

        return CalibrationResults(
            input_grid_2d=grid_2d,
            regression_targets=targets.detach().cpu().numpy(),
            mcmd_vals=mcmd_vals.detach().cpu().numpy(),
            mean_mcmd=mcmd_vals.mean().item(),
            ece=ece,
        )

    def compute_mcmd(
        self,
        model: DiscreteRegressionNN,
        data_loader: DataLoader,
        return_grid: bool = False,
        return_targets: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Compute the MCMD between samples drawn from the given model and the data spanned by the data loader.

        Args:
            model (DiscreteRegressionNN): Probabilistic regression model to compute the MCMD for.
            data_loader (DataLoader): DataLoader with the test data to compute MCMD over.
            return_grid (bool, optional): Whether/not to return the grid of values the MCMD was computed over. Defaults to False.
            return_targets (bool, optional): Whether/not to return the regression targets the MCMD was computed against. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The computed MCMD values, along with the grid of inputs these values correspond to (if return_grid is True) and the regression targets (if return_targets is True).
        """
        x, y, x_prime, y_prime = self._get_samples_for_mcmd(model, data_loader)
        x_kernel, y_kernel = self._get_kernel_functions(y)
        mcmd_vals = compute_mcmd_torch(
            grid=x,
            x=x,
            y=y,
            x_prime=x_prime,
            y_prime=y_prime,
            x_kernel=x_kernel,
            y_kernel=y_kernel,
            lmbda=self.settings.mcmd_lmbda,
        )
        return_obj = [mcmd_vals]
        if return_grid:
            return_obj.append(x)
        if return_targets:
            return_obj.append(y)
        if len(return_obj) == 1:
            return return_obj[0]
        else:
            return tuple(return_obj)

    def compute_ece(self, model: DiscreteRegressionNN, data_loader: DataLoader) -> float:
        """Compute the regression ECE of the given model over the dataset spanned by the data loader.

        Args:
            model (DiscreteRegressionNN): Probabilistic regression model to compute the ECE for.
            data_loader (DataLoader): DataLoader with the test data to compute ECE over.

        Returns:
            float: The regression ECE.
        """
        all_outputs = []
        all_targets = []
        for inputs, targets in tqdm(
            data_loader, desc="Getting posterior predictive dists for MCMD..."
        ):
            all_outputs.append(model.predict(inputs.to(model.device)))
            all_targets.append(targets.to(model.device))

        all_targets = torch.cat(all_targets).detach().cpu().numpy()
        all_outputs = torch.cat(all_outputs, dim=0)
        posterior_predictive = model.posterior_predictive(all_outputs)

        ece = compute_regression_ece(
            y_true=all_targets,
            posterior_predictive=posterior_predictive,
            num_bins=self.settings.ece_bins,
            weights=self.settings.ece_weights,
            alpha=self.settings.ece_alpha,
        )
        return ece

    def plot_mcmd_results(
        self,
        calibration_results: CalibrationResults,
        gridsize: int = 15,
        show: bool = False,
    ) -> plt.Figure:
        """Given a set of calibration results and an existing axes, plot the MCMD values against their 2d input projections on a hexbin grid.

        Args:
            calibration_results (CalibrationResults): Calibration results from a CalibrationEvaluator.
            ax (plt.Axes): The axes to draw the plot on.
            gridsize (int, optional): Gridsize parameter for the hexbin plot. Defaults to 15.
            show (bool, optional): Whether/not to show the resultant figure with plt.show(). Defaults to False.

        Returns:
            Figure: Matplotlib figure with the visualized MCMD values.
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs: Sequence[plt.Axes]
        axs[0].set_title("Data")
        hb1 = axs[0].hexbin(
            *calibration_results.input_grid_2d.T,
            calibration_results.regression_targets,
            gridsize=gridsize,
        )
        fig.colorbar(hb1, ax=axs[0])

        axs[1].set_title(f"Mean MCMD: {calibration_results.mean_mcmd:.4f}")
        hb2 = axs[1].hexbin(
            *calibration_results.input_grid_2d.T, calibration_results.mcmd_vals, gridsize=gridsize
        )
        fig.colorbar(hb2, ax=axs[1])

        fig.tight_layout()

        if show:
            plt.show()
        return fig

    def _get_samples_for_mcmd(
        self, model: DiscreteRegressionNN, data_loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = []
        y = []
        x_prime = []
        y_prime = []
        for inputs, targets in tqdm(data_loader, desc="Sampling from posteriors for MCMD..."):
            if self.settings.dataset_type == DatasetType.TABULAR:
                x.append(inputs)
            elif self.settings.dataset_type == DatasetType.IMAGE:
                x.append(self.clip_model.encode_image(inputs.to(self.device), normalize=True))
            elif self.settings.dataset_type == DatasetType.TEXT:
                x.append(self.clip_model.encode_text(inputs.to(self.device), normalize=True))
            y.append(targets.to(model.device))
            inputs = inputs.to(model.device)
            y_hat = model.predict(inputs)
            x_prime.append(torch.repeat_interleave(x[-1], repeats=5, dim=0))
            y_prime.append(
                model.sample(y_hat, num_samples=self.settings.mcmd_num_samples).flatten()
            )

        x = torch.cat(x, dim=0)
        y = torch.cat(y).float()
        x_prime = torch.cat(x_prime, dim=0)
        y_prime = torch.cat(y_prime).float()

        return x, y, x_prime, y_prime

    def _get_kernel_functions(self, y: torch.Tensor) -> tuple[KernelFunction, KernelFunction]:
        if self.settings.mcmd_input_kernel == "polynomial":
            x_kernel = polynomial_kernel
        else:
            x_kernel = self.settings.mcmd_input_kernel

        if self.settings.mcmd_output_kernel == "rbf":
            y_kernel = partial(rbf_kernel, gamma=(1 / (2 * y.float().var())))
        elif self.settings.mcmd_output_kernel == "laplacian":
            y_kernel = partial(laplacian_kernel, gamma=(1 / (2 * y.float().var())))

        return x_kernel, y_kernel

    @property
    def clip_model(self) -> CLIP:
        if self._clip_model is None:
            self._clip_model, _, self._image_preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                device=self.device,
            )
        return self._clip_model

    @property
    def image_preprocess(self) -> Compose:
        if self._image_preprocess is None:
            self._clip_model, _, self._image_preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                device=self.device,
            )
        return self._image_preprocess

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return self._tokenizer

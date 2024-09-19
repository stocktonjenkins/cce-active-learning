from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from scipy.stats import nbinom
from scipy.stats import norm
from scipy.stats import poisson
from sklearn.metrics.pairwise import rbf_kernel

from probcal.evaluation.metrics import compute_mcmd_numpy
from probcal.evaluation.plotting import plot_posterior_predictive
from probcal.models import GaussianNN
from probcal.models import NegBinomNN
from probcal.models import PoissonNN
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.utils.multiple_formatter import multiple_formatter


def produce_figure(
    models: list[DiscreteRegressionNN],
    names: list[str],
    save_path: Path | str,
    dataset_path: Path | str,
):
    """Create a figure showcasing the MCMD's ability to identify calibrated models.

    Args:
        models (list[DiscreteRegressionNN]): List of models to plot posterior predictive distributions of.
        names (list[str]): List of display names for each respective model in `models`.
        save_path (Path | str): Path to save figure to.
        dataset_path (Path | str): Path with dataset models were fit on.
    """
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, axs = plt.subplots(
        2,
        len(models),
        figsize=(2.5 * len(models), 4),
        sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    data: dict[str, np.ndarray] = np.load(dataset_path)
    X = data["X_test"].flatten()
    y = data["y_test"].flatten()
    num_posterior_samples = 100
    x_kernel = partial(rbf_kernel, gamma=0.5)
    y_kernel = partial(rbf_kernel, gamma=0.5)

    for i, (model, model_name) in enumerate(zip(models, names)):
        posterior_ax: plt.Axes = axs[0, i]
        mcmd_ax: plt.Axes = axs[1, i]

        if isinstance(model, GaussianNN):
            y_hat = model.predict(torch.tensor(X).unsqueeze(1))
            mu, var = torch.split(y_hat, [1, 1], dim=-1)
            mu = mu.flatten().detach().numpy()
            std = var.sqrt().flatten().detach().numpy()
            dist = norm(loc=mu, scale=std)
            nll = np.mean(-np.log(dist.cdf(y + 0.5) - dist.cdf(y - 0.5)))

        elif isinstance(model, PoissonNN):
            y_hat = model.predict(torch.tensor(X).unsqueeze(1))
            mu = y_hat.detach().numpy().flatten()
            dist = poisson(mu)
            nll = np.mean(-dist.logpmf(y))

        elif isinstance(model, NegBinomNN):
            y_hat = model.predict(torch.tensor(X).unsqueeze(1))
            mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
            mu = mu.flatten().detach().numpy()
            alpha = alpha.flatten().detach().numpy()

            eps = 1e-6
            var = mu + alpha * mu**2
            n = mu**2 / np.maximum(var - mu, eps)
            p = mu / np.maximum(var, eps)
            dist = nbinom(n=n, p=p)
            nll = np.mean(-dist.logpmf(y))

        pred = model.point_prediction(y_hat, training=False).detach().flatten().numpy()
        mae = np.abs(pred - y).mean()
        lower, upper = dist.ppf(0.025), dist.ppf(0.975)
        plot_posterior_predictive(
            X,
            y,
            mu,
            lower=lower,
            upper=upper,
            show=False,
            ax=posterior_ax,
            ylims=(0, 45),
            legend=False,
            error_color="gray",
        )
        posterior_ax.set_title(model_name)
        posterior_ax.annotate(f"MAE: {mae:.3f}", (0.2, 41))
        posterior_ax.annotate(f"NLL: {nll:.3f}", (0.2, 37))
        posterior_ax.xaxis.set_major_locator(MultipleLocator(np.pi))
        posterior_ax.xaxis.set_major_formatter(FuncFormatter(multiple_formatter()))
        posterior_ax.set_xlabel(None)
        posterior_ax.set_ylabel(None)

        mcmd_vals = compute_mcmd_numpy(
            grid=X,
            x=X,
            y=y,
            x_prime=np.tile(X, num_posterior_samples),
            y_prime=dist.rvs((num_posterior_samples, len(X))),
            x_kernel=x_kernel,
            y_kernel=y_kernel,
        )
        sorted_indices = np.argsort(X)
        mcmd_ax.plot(X[sorted_indices], mcmd_vals[sorted_indices])
        mcmd_ax.set_ylim(-0.01, 0.75)
        mcmd_ax.annotate(
            f"Mean MCMD: {np.mean(mcmd_vals):.4f}",
            (X.min() + 0.1, mcmd_ax.get_ylim()[1] * 0.8),
        )

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "probcal/figures/artifacts/mcmd_in_practice.pdf"
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    models = [
        PoissonNN.load_from_checkpoint("weights/discrete_sine_wave_poisson.ckpt"),
        NegBinomNN.load_from_checkpoint("weights/discrete_sine_wave_nbinom.ckpt"),
        GaussianNN.load_from_checkpoint("weights/discrete_sine_wave_gaussian.ckpt"),
    ]
    names = ["Poisson DNN", "NB DNN", "Gaussian DNN"]
    produce_figure(models, names, save_path, dataset_path)

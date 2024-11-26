from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import rv_continuous
from sklearn.metrics.pairwise import rbf_kernel

from probcal.evaluation.metrics import compute_mcmd_numpy
from probcal.evaluation.metrics import compute_regression_ece


def plot_regression_reliability_diagram(
    y_true: np.ndarray,
    posterior_predictive: rv_continuous,
    num_bins: int = 9,
    ax: plt.Axes | None = None,
    show: bool = True,
):
    """Given targets and a probabilistic regression model (represented as a continuous random variable over the targets), plot a reliability diagram.

    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive (rv_continuous): Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): Specifies how many probability thresholds to use for checking CDF calibration. This
                        corresponds to how many points will be plotted to form the calibration curve.
        ax (plt.Axes | None): The axis to plot on (if provided). If None is passed in, an axis is created.
        show (bool): Specifies whether/not to display the resultant plot.
    """
    epsilon = 1e-4
    p_vals = np.linspace(0 + epsilon, 1 - epsilon, num=num_bins).reshape(-1, 1)
    expected_pct_where_cdf_less_than_p = p_vals
    actual_pct_where_cdf_less_than_p = (
        posterior_predictive.cdf(y_true) <= p_vals
    ).mean(axis=1)

    ax = plt.subplots(1, 1)[1] if ax is None else ax
    ax.plot(
        expected_pct_where_cdf_less_than_p,
        expected_pct_where_cdf_less_than_p,
        linestyle="--",
        color="red",
        label="Perfectly calibrated",
    )
    ax.plot(
        expected_pct_where_cdf_less_than_p,
        actual_pct_where_cdf_less_than_p,
        marker="o",
        linestyle="-",
        color="black",
        label="Model",
        markersize=3,
    )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    if show:
        plt.show()


def generate_figure(alpha: float, save_dir: Path):
    x = np.random.normal(0, 1, size=500)
    y = np.random.normal(alpha * x, 1)

    predictive_dist = norm(loc=0, scale=np.sqrt(1 + alpha**2))
    x_prime = x
    y_prime = predictive_dist.rvs(size=len(x))
    support = np.linspace(-10, 10)

    ece = compute_regression_ece(
        y_true=y,
        posterior_predictive=predictive_dist,
        weights="frequency",
        num_bins=100,
    )
    mcmd_vals = compute_mcmd_numpy(
        grid=x,
        x=x,
        y=y,
        x_prime=x,
        y_prime=predictive_dist.rvs(len(x)),
        x_kernel=partial(rbf_kernel, gamma=1),
        y_kernel=partial(rbf_kernel, gamma=y.var() / 2),
        lmbda=0.1,
    )

    fig_a, subplot_a = plt.subplots(1, 1, figsize=(4, 4))
    fig_b, subplot_b = plt.subplots(1, 1, figsize=(4, 4))
    fig_c, subplot_c = plt.subplots(1, 1, figsize=(4, 4))
    fig_d, subplot_d = plt.subplots(1, 1, figsize=(4, 4))
    subplot_a: plt.Axes
    subplot_b: plt.Axes
    subplot_c: plt.Axes
    subplot_d: plt.Axes

    # Plot A: Show samples from the ground truth and "learned" distributions.
    subplot_a.scatter(x, y, label="$Y|X$", alpha=0.2, zorder=10, s=15)
    subplot_a.scatter(x_prime, y_prime, label="$f(x)$", alpha=0.2, s=15)
    subplot_a.legend()
    subplot_a.set_xlabel("$X$")
    subplot_a.set_ylabel("$Y$")
    fig_a.tight_layout()
    fig_a.savefig(save_dir / "calibration_flaws_a.pdf", dpi=150)

    # Plot B: Compare the "learned" (marginal) distribution to the histogram of the targets.
    subplot_b.hist(y, density=True, alpha=0.7, label="$Y|X$")
    subplot_b.plot(support, predictive_dist.pdf(support), alpha=0.8, label="f(x)")
    subplot_b.set_yticks([])
    subplot_b.set_xlabel("$Y$")
    subplot_b.set_ylabel("Density")
    subplot_b.legend()
    fig_b.tight_layout()
    fig_b.savefig(save_dir / "calibration_flaws_b.pdf", dpi=150)

    # Plot C: Plot the misleading "reliability" diagram of the model.
    subplot_c.annotate(f"ECE: {ece:.3f}", xy=(0.1, 0.8))
    plot_regression_reliability_diagram(
        y_true=y,
        posterior_predictive=predictive_dist,
        ax=subplot_c,
        show=False,
    )
    subplot_c.set_xlabel("$p$")
    subplot_c.set_ylabel(r"$\hat{p}$")
    fig_c.tight_layout()
    fig_c.savefig(save_dir / "calibration_flaws_c.pdf", dpi=150)

    # Plot D: Plot the MCMD values, which correctly show a large discrepancy in the distributions.
    order = np.argsort(x)
    subplot_d.plot(x[order], mcmd_vals[order])
    subplot_d.annotate(
        f"Avg. MCMD: {mcmd_vals.mean():.3f}", xy=(-2.5, mcmd_vals.max() * 1.2)
    )
    subplot_d.set_ylim(0, mcmd_vals.max() * 1.5)
    subplot_d.set_xlim(x.min(), x.max())
    subplot_d.set_xlabel("$X$")
    subplot_d.set_ylabel("MCMD")
    fig_d.tight_layout()
    fig_d.savefig(save_dir / "calibration_flaws_d.pdf", dpi=150)


if __name__ == "__main__":
    generate_figure(alpha=3, save_dir=Path("probcal/figures/artifacts"))

from functools import partial
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from sklearn.metrics.pairwise import rbf_kernel

from probcal.evaluation.metrics import compute_mcmd_numpy
from probcal.utils.multiple_formatter import multiple_formatter


def generate_homoscedastic_data(
    n: int = 1000,
    mean_modification: Literal["default", "higher", "lower"] = "default",
    var_modification: Literal["default", "higher", "lower"] = "default",
) -> tuple[np.ndarray, np.ndarray]:
    """Create a homoscedastic x-conditional dataset.

    Args:
        n (int, optional): The number of samples to draw. Defaults to 1000.
        mean_modification (str, optional): Specifies how to modify the mean when generating ("default", "higher", "lower"). Defaults to "default".
        var_modification (str, optional): Specifies how to modify the variance when generating ("default", "higher", "lower"). Defaults to "default".

    Returns:
        tuple[np.ndarray, np.ndarray]: The resultant dataset (returned in x, y order).
    """
    x = np.random.uniform(0, 2 * np.pi, size=n)

    if mean_modification == "default":
        mean = np.cos(x)
    elif mean_modification == "higher":
        mean = np.cos(x) + 2
    elif mean_modification == "lower":
        mean = np.cos(x) - 2

    if var_modification == "default":
        stdev = 0.5
    elif var_modification == "higher":
        stdev = 1.5
    elif var_modification == "lower":
        stdev = 0.1

    y = np.random.normal(loc=mean, scale=stdev)

    return x, y


def generate_figure(save_path: str):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    x, y = generate_homoscedastic_data(n=1000)
    grid = np.linspace(x.min(), x.max())
    x_same, y_same = generate_homoscedastic_data(
        n=500, mean_modification="default", var_modification="default"
    )
    x_same_mean_low_var, y_same_mean_low_var = generate_homoscedastic_data(
        n=500, mean_modification="default", var_modification="lower"
    )
    x_same_mean_high_var, y_same_mean_high_var = generate_homoscedastic_data(
        n=500, mean_modification="default", var_modification="higher"
    )
    x_low_mean_low_var, y_high_mean_low_var = generate_homoscedastic_data(
        n=500, mean_modification="lower", var_modification="lower"
    )
    x_low_mean_high_var, y_high_mean_high_var = generate_homoscedastic_data(
        n=500, mean_modification="lower", var_modification="higher"
    )
    x_diff_shape = np.random.uniform(x.min(), x.max(), size=500)
    y_diff_shape = np.random.normal(x_diff_shape - 5, 0.5)
    titles = [
        "$\mu\prime = \mu, \sigma \prime = \sigma$",
        "$\mu\prime = \mu, \sigma \prime < \sigma$",
        "$\mu\prime = \mu, \sigma \prime > \sigma$",
        "$\mu\prime \\neq \mu, \sigma \prime < \sigma$",
        "$\mu\prime \\neq \mu, \sigma \prime > \sigma$",
        "Different Shape",
    ]

    gamma = 0.5
    x_kernel = partial(rbf_kernel, gamma=gamma)
    y_kernel = partial(rbf_kernel, gamma=gamma)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=6,
        figsize=(12, 4),
        sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for i, ((x_prime, y_prime), title) in enumerate(
        zip(
            [
                (x_same, y_same),
                (x_same_mean_low_var, y_same_mean_low_var),
                (x_same_mean_high_var, y_same_mean_high_var),
                (x_low_mean_low_var, y_high_mean_low_var),
                (x_low_mean_high_var, y_high_mean_high_var),
                (x_diff_shape, y_diff_shape),
            ],
            titles,
        )
    ):
        scatter_ax: plt.Axes = axs[0, i]
        mcmd_ax: plt.Axes = axs[1, i]

        scatter_ax.set_title(title, fontsize=10)
        scatter_ax.scatter(x, y, alpha=0.5, s=5, label="$(x, y)$")
        scatter_ax.scatter(
            x_prime, y_prime, alpha=0.5, s=5, label="$(x\prime, y\prime)$"
        )
        scatter_ax.set_ylim(-7, 7)
        scatter_ax.xaxis.set_major_locator(MultipleLocator(np.pi))
        scatter_ax.xaxis.set_major_formatter(FuncFormatter(multiple_formatter()))
        mcmd_vals = compute_mcmd_numpy(grid, x, y, x_prime, y_prime, x_kernel, y_kernel)

        mcmd_ax.plot(grid, mcmd_vals)
        mcmd_ax.set_ylim(-0.1, 2.5)
        mcmd_ax.annotate(
            f"Mean MCMD: {np.mean(mcmd_vals):.4f}",
            (0.1, 0.8 * mcmd_ax.get_ylim()[1]),
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    save_path = "probcal/figures/artifacts/homoscedastic_behavior.pdf"
    generate_figure(save_path)

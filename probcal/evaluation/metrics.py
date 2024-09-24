from typing import Callable
from typing import TypeAlias

import numpy as np
import torch
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.stats import rv_continuous

from probcal.random_variables.discrete_random_variable import DiscreteRandomVariable


def _compute_discrete_torch_dist_cdf(
    dist: torch.distributions.Poisson | torch.distributions.NegativeBinomial,
    y_vals: torch.Tensor,
    max_val: int = 2000,
) -> torch.Tensor:
    support = torch.arange(max_val, device=y_vals.device).unsqueeze(0).repeat(len(y_vals), 1)
    probs_over_support: torch.Tensor = dist.log_prob(support).exp()
    probs_over_support *= support <= y_vals.view(-1, 1)
    cdf = probs_over_support.sum(dim=1)
    return cdf


def compute_regression_ece(
    y_true: np.ndarray,
    posterior_predictive: rv_continuous
    | DiscreteRandomVariable
    | torch.distributions.Distribution,
    num_bins: int = 100,
    weights: str = "uniform",
    alpha: float = 1.0,
) -> float:
    """Given targets and a probabilistic regression model (represented as a continuous random variable over the targets), compute the expected calibration error of the model.

    Given a set of probability values {p_1, ..., p_m} spanning [0, 1], and a set of regression targets {y_i | 1 <= i <= n}, the expected calibration error is defined as follows:

        If F_i denotes the posterior predictive cdf for y_i and q_j = |{y_i | F_i(y_i) <= p_j, i = 1, 2, ..., n}| / n, we have

            ECE = sum(w_j * abs(p_j - q_j)^alpha)

        where alpha controls the severity of penalty for a given probability residual.

    Args:
        y_true (np.ndarray): The true values of the regression targets.
        posterior_predictive (rv_continuous | DiscreteRandomVariable | torch.distributions.Distribution): Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): The number of bins to use for the ECE. Defaults to 100.
        weights (str, optional): Strategy for choosing the weights in the ECE sum. Must be either "uniform" or "frequency" (terms are weighted by the numerator of q_j). Defaults to "uniform".
        alpha (float, optional): Controls how severely we penalize the model for the distance between p_j and q_j. Defaults to 1 (error term is |p_j - q_j|).

    Returns:
        float: The expected calibration error.
    """
    TorchDistribution: TypeAlias = torch.distributions.Distribution | DiscreteRandomVariable
    eps = 1e-5
    p_j = np.linspace(eps, 1 - eps, num=num_bins)

    if isinstance(posterior_predictive, TorchDistribution):
        if isinstance(posterior_predictive, torch.distributions.Distribution):
            device = None
            for param in posterior_predictive.__dict__.values():
                if isinstance(param, torch.Tensor) and str(param.device) != "cpu":
                    device = param.device
            device = device or torch.device("cpu")
        elif isinstance(posterior_predictive, DiscreteRandomVariable):
            device = posterior_predictive.device
        y_true_torch = torch.tensor(y_true, device=device)
        p_j_torch = torch.tensor(p_j, device=device).reshape(-1, 1)

        # CDF is not currently implemented for the Poisson or Negative Binomial in PyTorch, so we approximate it.
        if isinstance(posterior_predictive, torch.distributions.Poisson):
            cdf = _compute_discrete_torch_dist_cdf(
                dist=torch.distributions.Poisson(posterior_predictive.rate.view(-1, 1)),
                y_vals=y_true_torch,
            )
        elif isinstance(posterior_predictive, torch.distributions.NegativeBinomial):
            cdf = _compute_discrete_torch_dist_cdf(
                dist=torch.distributions.NegativeBinomial(
                    total_count=posterior_predictive.total_count.view(-1, 1),
                    probs=posterior_predictive.probs.view(-1, 1),
                ),
                y_vals=y_true_torch,
            )
        else:
            cdf = posterior_predictive.cdf(y_true_torch.flatten().long())

        cdf_less_than_p = cdf <= p_j_torch
        cdf_less_than_p = cdf_less_than_p.detach().cpu().numpy()
    else:
        cdf_less_than_p = posterior_predictive.cdf(y_true) <= p_j.reshape(-1, 1)

    q_j = cdf_less_than_p.mean(axis=1)

    if weights == "uniform":
        w_j = np.ones_like(q_j)
    elif weights == "frequency":
        w_j = cdf_less_than_p.sum(axis=1)
    else:
        raise ValueError(
            f"Weights strategy must be either 'uniform' or 'frequency'. Received {weights}"
        )

    w_j = w_j / w_j.sum()
    ece = np.dot(w_j, np.abs(p_j - q_j) ** alpha)
    return ece


def compute_mcmd_numpy(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_prime: np.ndarray,
    y_prime: np.ndarray,
    x_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lmbda: float = 0.01,
) -> np.ndarray:
    """Given a ground-truth conditional distribution and samples from a model's approximation of that distribution, compute the maximum conditional mean discrepancy (MCMD) along the provided grid.

    Args:
        grid (np.ndarray): Grid of values (assumed to be drawn from X) to compute MCMD across. Shape: (k, d) or (k,)
        x (np.ndarray): The conditioning values that produced y. Shape: (n, d) or (n,).
        y (np.ndarray): Ground truth samples from the conditional distribution. Shape: (n, 1) or (n,).
        x_prime (np.ndarray): The conditioning values that produced y_prime. Shape: (m, d) or (m,).
        y_prime (np.ndarray): Samples from a model's approximation of the ground truth conditional distribution. Shape: (m, 1) or (m,).
        x_kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): Kernel function to use for the conditioning variable (x).
        y_kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): Kernel function to use for the output variable (y).
        lmbda (float, optional): Regularization parameter. Defaults to 0.01.

    Returns:
        np.ndarray: MCMD values along the provided grid. Shape: (k,).
    """
    if grid.ndim == 1:
        grid = grid.reshape(-1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x_prime.ndim == 1:
        x_prime = x_prime.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_prime.ndim == 1:
        y_prime = y_prime.reshape(-1, 1)

    n = len(x)
    m = len(x_prime)

    K_X = x_kernel(x, x)
    K_X_prime = x_kernel(x_prime, x_prime)

    W_X = cho_solve(cho_factor(K_X + n * lmbda * np.eye(n)), np.eye(n))
    W_X_prime = cho_solve(cho_factor(K_X_prime + m * lmbda * np.eye(m)), np.eye(m))

    K_Y = y_kernel(y, y)
    K_Y_prime = y_kernel(y_prime, y_prime)
    K_Y_Y_prime = y_kernel(y, y_prime)

    k_X = x_kernel(x, grid)
    k_X_prime = x_kernel(x_prime, grid)

    A_1 = W_X @ K_Y @ W_X.T
    A_2 = W_X @ K_Y_Y_prime @ W_X_prime.T
    A_3 = W_X_prime @ K_Y_prime @ W_X_prime.T

    path = ["einsum_path", (0, 1), (0, 1)]
    first_term = np.einsum("ij,jk,ki->i", k_X.T, A_1, k_X, optimize=path)
    second_term = 2 * np.einsum("ij,jk,ki->i", k_X.T, A_2, k_X_prime, optimize=path)
    third_term = np.einsum("ij,jk,ki->i", k_X_prime.T, A_3, k_X_prime, optimize=path)

    return first_term - second_term + third_term


def compute_mcmd_torch(
    grid: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    x_prime: torch.Tensor,
    y_prime: torch.Tensor,
    x_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lmbda: float = 0.01,
) -> torch.Tensor:
    """Given a ground-truth conditional distribution and samples from a model's approximation of that distribution, compute the maximum conditional mean discrepancy (MCMD) along the provided grid.

    This method works entirely in PyTorch (allowing GPU speedups where applicable).

    Args:
        grid (torch.Tensor): Grid of values (assumed to be drawn from X) to compute MCMD across. Shape: (k, d) or (k,).
        x (torch.Tensor): The conditioning values that produced y. Shape: (n, d) or (n,).
        y (torch.Tensor): Ground truth samples from the conditional distribution. Shape: (n, 1) or (n,).
        x_prime (torch.Tensor): The conditioning values that produced y_prime. Shape: (m, d) or (m,).
        y_prime (torch.Tensor): Samples from a model's approximation of the ground truth conditional distribution. Shape: (m, 1) or (m,).
        x_kernel (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Kernel function to use for the conditioning variable (x).
        y_kernel (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Kernel function to use for the output variable (y).
        lmbda (float, optional): Regularization parameter. Defaults to 0.01.

    Returns:
        torch.Tensor: MCMD values along the provided grid. Shape: (k,).
    """
    if grid.dim() == 1:
        grid = grid.reshape(-1, 1)
    if x.dim() == 1:
        x = x.reshape(-1, 1)
    if x_prime.dim() == 1:
        x_prime = x_prime.reshape(-1, 1)
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    if y_prime.dim() == 1:
        y_prime = y_prime.reshape(-1, 1)

    n = len(x)
    m = len(x_prime)
    device = x.device
    I_n = torch.eye(n, device=device)
    I_m = torch.eye(m, device=device)

    K_X = x_kernel(x, x)
    K_X_prime = x_kernel(x_prime, x_prime)

    L = torch.linalg.cholesky(K_X + n * lmbda * I_n)
    L_prime = torch.linalg.cholesky(K_X_prime + m * lmbda * I_m)
    W_X = torch.cholesky_inverse(L)
    W_X_prime = torch.cholesky_inverse(L_prime)

    K_Y = y_kernel(y, y)
    K_Y_prime = y_kernel(y_prime, y_prime)
    K_Y_Y_prime = y_kernel(y, y_prime)

    k_X = x_kernel(x, grid)
    k_X_prime = x_kernel(x_prime, grid)

    A_1 = W_X @ K_Y @ W_X.T
    A_2 = W_X @ K_Y_Y_prime @ W_X_prime.T
    A_3 = W_X_prime @ K_Y_prime @ W_X_prime.T

    first_term = torch.einsum("ij,jk,ki->i", k_X.T, A_1, k_X)
    second_term = 2 * torch.einsum("ij,jk,ki->i", k_X.T, A_2, k_X_prime)
    third_term = torch.einsum("ij,jk,ki->i", k_X_prime.T, A_3, k_X_prime)

    return first_term - second_term + third_term

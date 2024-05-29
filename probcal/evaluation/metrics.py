from typing import Callable

import numpy as np
from scipy.stats import rv_continuous


def compute_regression_ece(
    y_true: np.ndarray,
    posterior_predictive: rv_continuous,
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
        posterior_predictive (RandomVariable): Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): The number of bins to use for the ECE. Defaults to 100.
        weights (str, optional): Strategy for choosing the weights in the ECE sum. Must be either "uniform" or "frequency" (terms are weighted by the numerator of q_j). Defaults to "uniform".
        alpha (float, optional): Controls how severely we penalize the model for the distance between p_j and q_j. Defaults to 1 (error term is |p_j - q_j|).

    Returns:
        float: The expected calibration error.
    """
    eps = 1e-5
    p_j = np.linspace(eps, 1 - eps, num=num_bins)
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


def compute_mcmd(
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
        grid (np.ndarray): Grid of values (assumed to be drawn from X) to compute MCMD across.
        x (np.ndarray): The conditioning values that produced y.
        y (np.ndarray): Ground truth samples from the conditional distribution.
        x_prime (np.ndarray): The conditioning values that produced y_prime.
        y_prime (np.ndarray): Samples from a model's approximation of the ground truth conditional distribution.
        x_kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): Kernel function to use for the conditioning variable (x).
        y_kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): Kernel function to use for the output variable (y).
        lmbda (float, optional): Regularization parameter. Defaults to 0.01.

    Returns:
        np.ndarray: MCMD values along the provided grid.
    """
    n = len(x)
    m = len(x_prime)

    K_X = x_kernel(x.reshape(-1, 1), x.reshape(-1, 1))
    K_X_prime = x_kernel(x_prime.reshape(-1, 1), x_prime.reshape(-1, 1))

    W_X = np.linalg.inv(K_X + n * lmbda * np.eye(n))
    W_X_prime = np.linalg.inv(K_X_prime + m * lmbda * np.eye(m))

    K_Y = y_kernel(y.reshape(-1, 1), y.reshape(-1, 1))
    K_Y_prime = y_kernel(y_prime.reshape(-1, 1), y_prime.reshape(-1, 1))
    K_Y_Y_prime = y_kernel(y.reshape(-1, 1), y_prime.reshape(-1, 1))

    k_X = x_kernel(x.reshape(-1, 1), grid.reshape(-1, 1))
    k_X_prime = x_kernel(x_prime.reshape(-1, 1), grid.reshape(-1, 1))

    first_term = np.diag(k_X.T @ W_X @ K_Y @ W_X.T @ k_X)
    second_term = np.diag(2 * k_X.T @ W_X @ K_Y_Y_prime @ W_X_prime.T @ k_X_prime)
    third_term = np.diag(k_X_prime.T @ W_X_prime @ K_Y_prime @ W_X_prime.T @ k_X_prime)

    return first_term - second_term + third_term

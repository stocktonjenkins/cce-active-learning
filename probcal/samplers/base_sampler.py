from abc import ABC
from abc import abstractmethod
from typing import Union

import numpy as np
import scipy
import torch


class BaseSampler(ABC):
    def __init__(self, yhat: Union[torch.Tensor, np.ndarray], as_tensor: bool = True):
        self.n = yhat.shape[0]
        self.as_tensor = as_tensor
        self.dist = self.yhat_to_rvs(yhat)

    @abstractmethod
    def yhat_to_rvs(
        self, yhat: torch.Tensor
    ) -> Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete]:
        pass

    @abstractmethod
    def get_nll(self, samples):
        pass

    def sample(self, m: int) -> Union[np.ndarray, torch.Tensor]:
        """
        Draw samples from the underlying scipy rvs object.
        Args:
            m: (int) number of samples to draw

        Returns: (np.ndarray) samples of shape (m, n) where n is the number of model predictions (y_hat) and m is the
        number of somples for each model output

        """
        draws = self.dist.rvs(size=(m, self.n))
        if self.as_tensor:
            draws = torch.tensor(draws)
        return draws

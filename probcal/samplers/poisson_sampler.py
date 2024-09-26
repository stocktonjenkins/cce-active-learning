import numpy as np
import scipy
import torch

from probcal.samplers.base_sampler import BaseSampler


class PoissonSampler(BaseSampler):
    def __init__(self, yhat: torch.Tensor):
        super(PoissonSampler, self).__init__(yhat)

    def yhat_to_rvs(self, yhat):
        mu = yhat.detach().numpy().flatten()
        dist = scipy.stats.poisson(mu)
        return dist

    def get_nll(self, samples):
        nll = np.mean(-self.dist.logpmf(samples))
        return nll

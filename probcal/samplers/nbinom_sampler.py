import numpy as np
import scipy
import torch

from probcal.samplers.base_sampler import BaseSampler


class NegBinomSampler(BaseSampler):
    def __init__(self, yhat: torch.Tensor):
        super(NegBinomSampler, self).__init__(yhat)

    def yhat_to_rvs(self, yhat):
        mu, alpha = torch.split(yhat, [1, 1], dim=-1)
        mu = mu.flatten().detach().numpy()
        alpha = alpha.flatten().detach().numpy()

        eps = 1e-6
        var = mu + alpha * mu**2
        n = mu**2 / np.maximum(var - mu, eps)
        p = mu / np.maximum(var, eps)
        dist = scipy.stats.nbinom(n=n, p=p)

        return dist

    def get_nll(self, samples):
        nll = np.mean(-self.dist.logpmf(samples))
        return nll

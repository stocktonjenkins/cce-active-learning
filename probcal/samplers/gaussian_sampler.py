import scipy
import torch

from probcal.samplers.base_sampler import BaseSampler


class GaussianSampler(BaseSampler):
    def __init__(self, yhat: torch.Tensor):
        super(GaussianSampler, self).__init__(yhat)

    def yhat_to_rvs(self, yhat):
        mu, var = torch.split(yhat, [1, 1], dim=-1)
        mu = mu.flatten().detach().numpy()
        std = var.sqrt().flatten().detach().numpy()
        return scipy.stats.norm(loc=mu, scale=std)

    def get_nll(self, samples):
        return -self.dist.logpdf(samples)

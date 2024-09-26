import torch


class MixUpTransform:
    def __init__(self, lmbda=1.0):
        assert 0 <= lmbda <= 1, "Lambda must be between 0 and 1."
        self.lmbda = lmbda

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (1 - self.lmbda) * x1 + self.lmbda * x2


class GaussianNoiseTransform:
    def __init__(self, mean: float, std: float):
        self.mean: torch.Tensor = torch.Tensor([mean])
        self.std: torch.Tensor = torch.Tensor([std])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.normal(mean=self.mean, std=self.std)

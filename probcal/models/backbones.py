import torch
from torch import nn


class Backbone(nn.Module):
    """Base class to ensure a common interface for all backbones.

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(Backbone, self).__init__()
        self.output_dim = output_dim


class MLP(Backbone):
    """An MLP feature extractor for (N, d) input data.

    Attributes:
        layers (nn.Sequential): The layers of this MLP.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 64):
        """Instantiate an MLP backbone.

        Args:
            input_dim (int, optional): Dimension of input feature vectors. Defaults to 1.
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        self.input_dim = input_dim
        super(MLP, self).__init__(output_dim=output_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

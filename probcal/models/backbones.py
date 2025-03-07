import torch
import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v3_large
from torchvision.models.mobilenet import MobileNet_V3_Large_Weights
import torchvision.models as tvmodels

from transformers import BatchEncoding
from transformers import DistilBertModel
from transformers import ViTModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling


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


class LargerMLP(Backbone):
    """A larger MLP feature extractor for (N, d) input data.

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
        super(LargerMLP, self).__init__(output_dim=output_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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


class MNISTCNN(Backbone):
    """A CNN feature extractor for the MNIST dataset (1x28x28 image tensors).

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(MNISTCNN, self).__init__(output_dim=output_dim)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(5 * 5 * 64, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.max_pool(self.relu(self.conv1(x))))
        x = self.bn2(self.max_pool(self.relu(self.conv2(x))))
        x = self.dropout(self.flat(x))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x


class ViT(Backbone):
    """A ViT feature extractor for images.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a ViT image feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(ViT, self).__init__(output_dim=output_dim)
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.projection_1 = nn.Linear(768, 384)
        self.projection_2 = nn.Linear(384, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: BaseModelOutputWithPooling = self.backbone(pixel_values=x)
        h = outputs.pooler_output
        h = self.relu(self.projection_1(h))
        h = self.relu(self.projection_2(h))
        return h


class ResNet18(Backbone):
    """A ResNet18 feature extractor for 3x224x224 image tensors.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a ResNet18 feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(ResNet18, self).__init__(output_dim=output_dim)

        self.backbone = tvmodels.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(
            in_features=512, out_features=output_dim, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h


class ResNet34(Backbone):
    """A ResNet34 feature extractor for 3x224x224 image tensors.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a ResNet34 feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(ResNet34, self).__init__(output_dim=output_dim)

        self.backbone = tvmodels.resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(
            in_features=512, out_features=output_dim, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h


class ResNet50(Backbone):
    """A ResNet50 feature extractor for 3x224x224 image tensors.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a ResNet50 feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(ResNet50, self).__init__(output_dim=output_dim)

        self.backbone = tvmodels.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(
            in_features=2048, out_features=output_dim, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h


class MobileNetV3(Backbone):
    """A MobileNetV3 feature extractor for 3x224x224 image tensors.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a MobileNetV3 feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(MobileNetV3, self).__init__(output_dim=output_dim)

        self.backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT
        ).features
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=2)
        self.conv1d = nn.Conv1d(
            in_channels=960, out_channels=self.output_dim, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(self.avg_pool(self.backbone(x)))
        h = self.conv1d(h).squeeze(-1)
        return h


class DistilBert(Backbone):
    """A DistilBert feature extractor for text sequences.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a DistilBert text feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(DistilBert, self).__init__(output_dim=output_dim)
        self.backbone = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.projection_1 = nn.Linear(768, 384)
        self.projection_2 = nn.Linear(384, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        outputs: BaseModelOutput = self.backbone(**x)
        h = outputs.last_hidden_state[:, 0]
        h = self.relu(self.projection_1(h))
        h = self.relu(self.projection_2(h))
        return h

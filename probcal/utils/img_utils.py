import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import torchvision.transforms.functional as F

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def img_show(imgs: List[torch.Tensor]):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = img.permute(1, 2, 0)
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def denormalize(tensor):
    # Clone the tensor so the original stays unmodified
    tensor = tensor.clone()

    # De-normalize by multiplying by the std and then adding the mean
    for t, m, s in zip(tensor, IMAGE_NET_MEAN, IMAGE_NET_STD):
        t.mul_(s).add_(m)

    return tensor

from torchvision.utils import make_grid
from torchvision.transforms import GaussianBlur

from probcal.utils.experiment_utils import get_model, get_datamodule
from probcal.utils.img_utils import denormalize
from probcal.enums import DatasetType, ImageDatasetName, HeadType

import matplotlib.pyplot as plt


datamodule = get_datamodule(
        DatasetType.IMAGE,
        ImageDatasetName.COCO_PEOPLE,
        1,
        num_workers=0
    )
#datamodule.prepare_data()
datamodule.setup(
    stage="test",
    transform=[GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5.0))]
)

test_loader = datamodule.test_dataloader()

imgs_to_show = []

for i, (x, y) in enumerate(test_loader):
    print(x.shape, y.shape)
    img = denormalize(x)
    img = img.squeeze(0).permute(1, 2, 0).detach()
    plt.imshow(img)
    plt.show()
    imgs_to_show.append(x.squeeze(0))
    if i == 3:
        break


import matplotlib.pyplot as plt

from probcal.enums import DatasetType
from probcal.enums import HeadType
from probcal.enums import ImageDatasetName
from probcal.utils.configs import TestConfig
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


# build dataset and data loader
datamodule = get_datamodule(DatasetType.IMAGE, ImageDatasetName.OOD_COCO_PEOPLE, 1, num_workers=0)
datamodule.setup(stage="test")
test_loader = datamodule.test_dataloader()

# instantiate model
model_cfg = TestConfig.from_yaml(f"configs/test/coco_gaussian_cfg.yaml")
model, intializer = get_model(model_cfg, return_initializer=True)

model = intializer.load_from_checkpoint(chkp_path)


imgs_to_show = []

for i, (x, y) in enumerate(test_loader):
    print(x.shape, y.shape)
    img = datamodule.denormalize(x)
    img = img.squeeze(0).permute(1, 2, 0).detach()
    plt.imshow(img)
    plt.show()
    imgs_to_show.append(x.squeeze(0))
    if i == 3:
        break

from probcal.utils.experiment_utils import get_model, get_datamodule

from probcal.enums import DatasetType, ImageDatasetName, HeadType

datamodule = get_datamodule(
        DatasetType.IMAGE,
        ImageDatasetName.COCO_PEOPLE,
        1,
        num_workers=0
    )
datamodule.prepare_data()
datamodule.setup("test")

test_loader = datamodule.test_dataloader()

for i, (x, y) in enumerate(test_loader):
    print(x.shape, y.shape)
    if i == 0:
        break
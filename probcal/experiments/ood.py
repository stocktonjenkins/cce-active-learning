from probcal.utils.experiment_utils import get_model, get_datamodule

from probcal.enums import DatasetType, ImageDatasetName, HeadType

datamodule = get_datamodule(
        DatasetType.IMAGE,
        ImageDatasetName.COCO_PEOPLE,
        1
    )



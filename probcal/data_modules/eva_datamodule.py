from pathlib import Path

from probcal.custom_datasets import EVADataset
from probcal.data_modules.prob_cal_data_module import ProbCalDataModule


class EVADataModule(ProbCalDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        self.root_dir = root_dir
        super().__init__(
            full_dataset=EVADataset(
                root_dir,
                surface_image_path=surface_image_path,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def prepare_data(self) -> None:
        # download and extract the data
        EVADataset(self.root_dir)

    @classmethod
    def denormalize(cls, tensor):
        # Clone the tensor so the original stays unmodified
        tensor = tensor.clone()

        # De-normalize by multiplying by the std and then adding the mean
        for t, m, s in zip(tensor, cls.IMAGE_NET_MEAN, cls.IMAGE_NET_STD):
            t.mul_(s).add_(m)

        return tensor

from pathlib import Path

from probcal.custom_datasets import AAFDataset
from probcal.data_modules.prob_cal_data_module import ProbCalDataModule


class AAFDataModule(ProbCalDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(
            full_dataset=AAFDataset(
                root_dir,
                surface_image_path=surface_image_path,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

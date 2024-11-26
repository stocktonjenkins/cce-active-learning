from pathlib import Path

from probcal.custom_datasets import FGNetDataset

from probcal.data_modules.prob_cal_data_module import ProbCalDataModule


class FGNetDataModule(ProbCalDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        ignore_grayscale: bool = True,
        surface_image_path: bool = False,
    ):
        self.root_dir = root_dir
        super().__init__(
            full_dataset=FGNetDataset(
                root_dir=root_dir,
                ignore_grayscale=ignore_grayscale,
                surface_image_path=surface_image_path,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        self.ignore_grayscale = ignore_grayscale

    def prepare_data(self) -> None:
        # Force images to be downloaded.
        FGNetDataset(self.root_dir)

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.lib.observer import IObserver, ISubject


class ActiveLearningAverageCCELogger(IObserver):
    def __init__(self, path: Path | str):
        self.path = path
        lock_path = path.split(".")[0] + ".lock"
        self.lock_path = lock_path
        with FileLock(self.lock_path):
            with open(self.path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Kth Trial",
                        "A.L. Iteration",
                        "Training",
                        "Validation",
                        "Test",
                        "Unlabeled",
                        "Mean MCMD",
                        "ECE",
                    ]
                )

    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        state = subject.get_state()
        assert state.calibration_results is not None, "Calibration results not saved!"
        with FileLock(self.lock_path):
            df = pd.read_csv(self.path)
            df.loc[len(df)] = [
                state.kth_trial,
                state.iteration,
                state.train_set_size,
                state.val_set_size,
                state.test_set_size,
                state.unlabeled_set_size,
                np.concatenate(
                    [vals.mcmd_vals for vals in state.calibration_results.mcmd_results]
                ).mean(),
                state.calibration_results.ece,
            ]
            df = df.round(
                {col: 4 for col in df.select_dtypes(include=["float"]).columns}
            )
            df.drop_duplicates(inplace=True)
            df.to_csv(self.path, index=False)

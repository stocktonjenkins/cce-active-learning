import csv
from pathlib import Path

import numpy as np
from filelock import FileLock

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.lib.observer import IObserver, ISubject


class ActiveLearningAverageCCELogger(IObserver):
    def __init__(self, path: Path | str):
        self.path = path
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
        with FileLock(self.path):
            with open(self.path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        state.kth_trial,
                        state.iteration,
                        state.train_set_size,
                        state.val_set_size,
                        state.test_set_size,
                        state.unlabeled_set_size,
                        np.concatenate(
                            [
                                vals.mcmd_vals
                                for vals in state.calibration_results.mcmd_results
                            ]
                        ).mean(),
                        state.calibration_results.ece,
                    ]
                )

import csv
from pathlib import Path

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
                    "A.L. Iteration",
                    "Training Set Size",
                    "Val. Set Size",
                    "Unlabeled Set Size",
                    # TODO: other columns?
                ]
            )

    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        # do something with subject.get_state()
        print(subject.get_state().calibration_results)

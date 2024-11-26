from pathlib import Path

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.lib.observer import IObserver, ISubject
import csv


class ActiveLearningModelAccuracyLogger(IObserver[ActiveLearningEvaluationResults]):
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
                    "MAE",
                    "RMSE",
                    "LOSS",
                    "NLL",
                ]
            )

    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        state = subject.get_state()
        # Log the results to the CSV file
        with open(self.path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    state.iteration,
                    state.train_set_size,
                    state.val_set_size,
                    state.unlabeled_set_size,
                    state.model_accuracy_results.test_mae,
                    state.model_accuracy_results.test_rmse,
                    state.model_accuracy_results.test_loss,
                    state.model_accuracy_results.nll,
                ]
            )

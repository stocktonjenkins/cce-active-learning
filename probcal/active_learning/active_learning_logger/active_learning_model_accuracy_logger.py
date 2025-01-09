from pathlib import Path

import pandas as pd
from filelock import FileLock

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.lib.observer import IObserver, ISubject
import csv
import wandb
from lightning.pytorch.loggers import WandbLogger


class ActiveLearningModelAccuracyLogger(IObserver[ActiveLearningEvaluationResults]):
    def __init__(
        self, path: Path | str, wandb_logger: WandbLogger, logging: bool = True
    ):
        self.path = path
        lock_path = path.split(".")[0] + ".lock"
        self.lock_path = lock_path
        self.logging = logging
        self.wandb_logger = wandb_logger
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
                        "MAE",
                        "RMSE",
                        "LOSS",
                        "NLL",
                    ]
                )

    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        state = subject.get_state()
        if self.logging:
            metrics = {
                "active_learning/kth_trial": state.kth_trial,
                "active_learning/iteration": state.iteration,
                "dataset_sizes/train": state.train_set_size,
                "dataset_sizes/val": state.val_set_size,
                "dataset_sizes/test": state.test_set_size,
                "dataset_sizes/unlabeled": state.unlabeled_set_size,
                "metrics/mae": state.model_accuracy_results.test_mae,
                "metrics/rmse": state.model_accuracy_results.test_rmse,
                "metrics/loss": state.model_accuracy_results.test_loss,
                "metrics/nll": state.model_accuracy_results.nll,
            }

            # Log to WandB using the parent logger
            self.wandb_logger.log_metrics(metrics)
            print(f"Logged metrics to WandB: {metrics}")
        # Log the results to the CSV file
        with FileLock(self.lock_path):
            df = pd.read_csv(self.path)
            df.loc[len(df)] = [
                state.kth_trial,
                state.iteration,
                state.train_set_size,
                state.val_set_size,
                state.test_set_size,
                state.unlabeled_set_size,
                state.model_accuracy_results.test_mae,
                state.model_accuracy_results.test_rmse,
                state.model_accuracy_results.test_loss,
                state.model_accuracy_results.nll,
            ]
            df = df.round(
                {col: 4 for col in df.select_dtypes(include=["float"]).columns}
            )
            df.drop_duplicates(inplace=True)
            df.to_csv(self.path, index=False)

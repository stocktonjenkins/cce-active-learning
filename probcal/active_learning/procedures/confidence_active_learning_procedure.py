import lightning
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
    ModelAccuracyResults,
)
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.evaluation.calibration_evaluator import CalibrationResults
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class ConfidenceProcedure(ActiveLearningProcedure[ActiveLearningProcedure]):
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: DiscreteRegressionNN
    ) -> np.ndarray:
        """
        Randomly choose the next set of indices to add to label set
        Args:
            unlabeled_indices: np.ndarray
            k: int
            model: DiscreteRegressionNN

        Returns:
            A random subset of unlabeled indices.
        """
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()
        confindence_unlabelled = self.get_confidence_score_from_model(
            model,
            data_loader=unlabeled_dataloader,
        )
        assert confindence_unlabelled.shape[0] == len(unlabeled_indices)
        _, sampling_indices = torch.topk(confindence_unlabelled, k)

        return unlabeled_indices[sampling_indices]

    def _eval_impl(
        self, trainer: lightning.Trainer, model: DiscreteRegressionNN
    ) -> ActiveLearningEvaluationResults:
        calibration_results: CalibrationResults = self.cal_evaluator(
            model, data_module=self.dataset
        )
        results = trainer.test(model, datamodule=self.dataset)
        model_accuracy_results = ModelAccuracyResults(**results[0])
        return ActiveLearningEvaluationResults(
            calibration_results=calibration_results,
            model_accuracy_results=model_accuracy_results,
            iteration=self._iteration,
            train_set_size=self.dataset.train_indices.shape[0],
            val_set_size=self.dataset.val_indices.shape[0],
            unlabeled_set_size=self.dataset.unlabeled_indices.shape[0],
        )

    def get_confidence_score_from_model(
        self, model: DiscreteRegressionNN, data_loader: DataLoader
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            conf = []
            for inputs, _ in tqdm(
                data_loader, desc="doing forward pass to compute confidence..."
            ):
                y_hat = model.predict(inputs.to(model.device))
                (mu, var) = torch.split(y_hat, [1, 1], dim=-1)
                conf.append(var.flatten())

            conf = torch.cat(conf).float()

            return conf

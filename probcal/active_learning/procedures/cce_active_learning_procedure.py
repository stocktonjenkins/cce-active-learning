import lightning
import numpy as np
import torch
from torchmetrics.functional import precision

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
    ModelAccuracyResults,
)
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.evaluation.calibration_evaluator import CalibrationResults
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class CCEProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def get_next_label_set(
        self,
        unlabeled_indices: np.ndarray,
        k: int,
        model: DiscreteRegressionNN,
    ) -> np.ndarray:
        """
        Randomly choose the next set of indices to add to label set
        Args:
            unlabeled_indices: np.ndarray
            k: int
            model: the model used to compute CCE

        Returns:
            A random subset of unlabeled indices.
        """
        model = model.to("cpu")
        train_dataloader = self.dataset.train_dataloader()
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()
        cce_unlabeled = self.cal_evaluator.compute_mcmd_unlabeled(
            model,
            unlabeled_data_loader=unlabeled_dataloader,
            data_loader=train_dataloader,
        )
        assert cce_unlabeled.shape[0] == len(unlabeled_indices)
        _, sampling_indices = torch.topk(cce_unlabeled, k)

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

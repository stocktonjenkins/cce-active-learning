import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import lightning

from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.active_learning_types import (
    ModelAccuracyResults,
    ActiveLearningEvaluationResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.active_learning.procedures.base import (
    ActiveLearningProcedure,
)
from probcal.evaluation.calibration_evaluator import CalibrationResults


class BAITProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def get_next_label_set(self, model: DiscreteRegressionNN, unlabeled_indices: np.ndarray, k: int) -> np.ndarray:
        """
        Choose the next set of indices to add to the label set based on Fisher Information.
        
        Args:
            model: DiscreteRegressionNN
            unlabeled_indices: np.ndarray
            k: int

        Returns:
            A subset of unlabeled indices selected based on Fisher Information.
        """
        model.eval()
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()
        
        fisher_scores = []
        
        with torch.no_grad():
            for data, _ in unlabeled_dataloader:
                data = data.to(model.device)
                predictions = model(data)                
                for pred, input_data in zip(predictions, data):
                    model.zero_grad()
                    pred.backward(retain_graph=True)
                    fisher_score = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            fisher_score += torch.sum(param.grad**2).item()
                    fisher_scores.append(fisher_score)
        
        fisher_scores = torch.tensor(fisher_scores)
        assert fisher_scores.shape[0] == len(unlabeled_indices)
        
        _, sampling_indices = torch.topk(fisher_scores, k)
        print(len(sampling_indices))
        return unlabeled_indices[sampling_indices.numpy()]

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

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L

from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.active_learning_types import (
    ModelAccuracyResults,
    ActiveLearningEvaluationResults,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


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
        # Set the model to evaluation mode
        model.eval()
        
        # Create a DataLoader for the unlabeled data
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()

        # Compute Fisher Information scores for all unlabeled data
        fisher_scores = self.compute_fisher_information(model, unlabeled_dataloader)

        # Ensure the Fisher Information scores match the number of unlabeled indices
        assert fisher_scores.shape[0] == len(unlabeled_indices), \
            "Fisher Information scores must match the number of unlabeled indices."

        # Select the top-k indices with the highest Fisher Information
        _, top_indices = torch.topk(fisher_scores, k)

        return unlabeled_indices[top_indices.numpy()]

    def compute_fisher_information(self, model: DiscreteRegressionNN, dataloader: DataLoader) -> torch.Tensor:
        """
        Compute the Fisher Information for the unlabeled data.
        
        Args:
            model: DiscreteRegressionNN
            dataloader: DataLoader

        Returns:
            A tensor of Fisher Information scores for the unlabeled data.
        """
        fisher_information = []
        i=0
        print(len(dataloader))
        for data, _ in dataloader:
            data = data.to(model.device)
            data.requires_grad = True  # Ensure the input tensor requires gradients
            i+=1
            print(i)
            # Forward pass to get predictions
            predictions = model(data)

            # Compute Fisher Information for each sample in the batch
            batch_fisher_scores = []
            for pred in predictions:
                model.zero_grad()  # Clear previous gradients
                
                # Reduce the prediction to a scalar before calling backward()
                pred.sum().backward(retain_graph=True)

                # Sum of squared gradients as Fisher Information score
                fisher_score = sum((param.grad ** 2).sum().item() for param in model.parameters() if param.grad is not None)
                batch_fisher_scores.append(fisher_score)

            fisher_information.extend(batch_fisher_scores)

        return torch.tensor(fisher_information)

    def _eval_impl(self, trainer: L.Trainer, model: DiscreteRegressionNN) -> ActiveLearningEvaluationResults:
        """
        Evaluate the model and return the results, including calibration results and model accuracy results.
        
        Args:
            trainer: L.Trainer
            model: DiscreteRegressionNN

        Returns:
            ActiveLearningEvaluationResults
        """
        # Evaluate calibration results using the provided calibration evaluator
        calibration_results = self.cal_evaluator(model, data_module=self.dataset)
        
        # Use the trainer to test the model
        test_results = trainer.test(model, datamodule=self.dataset)
        model_accuracy_results = ModelAccuracyResults(**test_results[0])

        # Compile evaluation results
        return ActiveLearningEvaluationResults(
            calibration_results=calibration_results,
            model_accuracy_results=model_accuracy_results,
            iteration=self._iteration,
            train_set_size=len(self.dataset.train_indices),
            val_set_size=len(self.dataset.val_indices),
            unlabeled_set_size=len(self.dataset.unlabeled_indices),
        )

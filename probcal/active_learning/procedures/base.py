from abc import ABC
from typing import TypeVar, Union, Any

import lightning
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
    IActiveLearningDataModuleDelegate,
    ModelAccuracyResults,
)
from probcal.active_learning.procedures.utils import seed_torch
from probcal.data_modules.active_learning_data_module import ActiveLearningDataModule
from probcal.evaluation import CalibrationEvaluator
from probcal.lib.observer import Subject
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


T = TypeVar("T")
EvalState = Union[ActiveLearningEvaluationResults, T]


class ActiveLearningProcedure(
    Subject[EvalState],
    IActiveLearningDataModuleDelegate,
    ABC,
):
    dataset: ActiveLearningDataModule
    config: ActiveLearningConfig
    _k: int = 0
    _al_iteration: int = 0

    def __init__(self, dataset: ActiveLearningDataModule, config: ActiveLearningConfig):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.cal_evaluator = CalibrationEvaluator(config.settings)

    def eval(
        self,
        trainer: lightning.Trainer,
        model: DiscreteRegressionNN | None = None,
        best_path: str | None = None,
    ):
        """
        Evaluate the given model in this active learning procedure.
            - Model accuracy
            - Model calibration?
            - Other?
        Also updates the state to send to observers.
        Args:
            trainer: the lightning trainer used to train the model.
            model: the regression model to evaluate OR
            best_path: the path of the best model to evaluate
        Returns:

        """
        assert (
            model is not None or best_path is not None
        ), "`model` or `best_path` must be defined."
        results = trainer.test(
            model=model, ckpt_path=best_path, datamodule=self.dataset
        )
        model_accuracy_results = ModelAccuracyResults(**results[0])
        eval_dict = {
            "kth_trial": self._k,
            "model_accuracy_results": model_accuracy_results,
            "iteration": self._al_iteration,
            "train_set_size": self.dataset.train_indices.shape[0],
            "val_set_size": self.dataset.val_indices.shape[0],
            "test_set_size": self.dataset.test_indices.shape[0],
            "unlabeled_set_size": self.dataset.unlabeled_indices.shape[0],
        }
        if self.config.measure_calibration:
            calibration_results = self.cal_evaluator(model, data_module=self.dataset)
            eval_dict["calibration_results"] = calibration_results
        eval_cls, rest = self._eval_ext(trainer, model, best_path)
        evaluation: EvalState = eval_cls(**{**eval_dict, **rest})
        self.update_state(evaluation)

    def _eval_ext(
        self,
        trainer: lightning.Trainer,
        model: DiscreteRegressionNN | None = None,
        best_path: str | None = None,
    ) -> tuple[type[EvalState], dict[str, Any]]:
        """
        Adds evaluation stats and defines the type of the evaluation result.
        It must extend `ActiveLearningEvaluationResults`
        """
        return ActiveLearningEvaluationResults, {}

    def step(self, model: DiscreteRegressionNN):
        """
        Update `self.dataset` to include pool the unlabeled samples
        from AL into the training pool
        Returns:
            None
        """
        self.dataset.step(self, model)
        self._al_iteration += 1
        self.notify()

    def jump(self, seed: int):
        """
        Jump to the next trial in the experiment (of k runs), using the given seed.
        """
        self._al_iteration = 0
        self._k += 1
        self.dataset.reset(seed)
        seed_torch(seed, self.config.deterministic_settings)

    def update_state(self, evaluation: EvalState):
        """
        Update the subject state. Prep for notifying observers
        Args:
            Evaluation results of the active learning procedure
        """
        self._state = evaluation

    @staticmethod
    def get_embedding(model: DiscreteRegressionNN, loader: DataLoader) -> Tensor:
        """
        Returns the embedding of the given model by extracting last layer representations
        """
        embedding = []
        model.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(loader):
                emb = model.get_last_layer_representation(inputs)
                embedding.append(emb.data.cpu())
        return torch.cat(embedding, dim=0)

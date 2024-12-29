import numpy as np

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.procedures.lcmd.feature_data import TensorFeatureData
from probcal.active_learning.procedures.lcmd.feature_maps import IdentityFeatureMap
from probcal.active_learning.procedures.lcmd.features import Features
from probcal.active_learning.procedures.lcmd.selection import (
    LargestClusterMaxDistSelectionMethod,
)
from probcal.models.discrete_regression_nn import DiscreteRegressionNN


class LCMDProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: DiscreteRegressionNN
    ) -> np.ndarray:
        # TODO: use self.dataset.full_dataset to get embeddings from CLIP?
        #       how would this work for text?
        self.dataset.full_dataset
        X = TensorFeatureData(
            data=[...]  # TODO: tensor of (n_samples, n_features)
        )
        feature_data = {
            "train": X[self.dataset.train_indices],
            "pool": X[unlabeled_indices],
        }
        feature_map = IdentityFeatureMap(
            n_features=feature_data["train"].get_tensor(0).shape[-1]
        )
        features = {
            key: Features(feature_map, data) for key, data in feature_data.items()
        }
        alg = LargestClusterMaxDistSelectionMethod(
            pool_features=features["pool"],
            train_features=features["train"],
        )
        selected_indices = alg.select(batch_size=1).detach().cpu().numpy()
        return unlabeled_indices[selected_indices]

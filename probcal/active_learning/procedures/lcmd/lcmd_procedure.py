import numpy as np

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.active_learning.procedures.lcmd.feature_data import (
    TensorFeatureData,
)
from probcal.active_learning.procedures.lcmd.feature_maps import IdentityFeatureMap
from probcal.active_learning.procedures.lcmd.features import Features
from probcal.active_learning.procedures.lcmd.selection import (
    LargestClusterMaxDistSelectionMethod,
)
from probcal.models.regression_nn import RegressionNN


class LCMDProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def get_next_label_set(
        self, unlabeled_indices: np.ndarray, k: int, model: RegressionNN
    ) -> np.ndarray:
        feature_data = self.get_tensor_features(model)
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
        selected_indices = (
            # The select method returns indices of the "pool" set (unlabeled set)
            alg.select(batch_size=min(k, unlabeled_indices.shape[0]))
            .detach()
            .cpu()
            .numpy()
        )
        return unlabeled_indices[selected_indices]

    def get_tensor_features(self, model: RegressionNN) -> dict[str, TensorFeatureData]:
        train = self.get_embedding(model, loader=self.dataset.train_dataloader())
        pool = self.get_embedding(model, loader=self.dataset.unlabeled_dataloader())
        return {
            "pool": TensorFeatureData(pool),
            "train": TensorFeatureData(train),
        }

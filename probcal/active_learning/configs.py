from pathlib import Path

import torch
from zmq.backend.cffi import device

from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.utils.generic_utils import get_yaml


class ActiveLearningConfig:
    @staticmethod
    def from_yaml(config_path: str | Path) -> "ActiveLearningConfig":
        config_dict = get_yaml(config_path)
        config_dict["settings"] = CalibrationEvaluatorSettings(
            **config_dict["settings"],
            device=torch.device("cpu")
        )

        return ActiveLearningConfig(**config_dict)

    def __init__(
        self,
        num_iter: int,
        label_k: int,
        init_num_labeled: int,
        settings: CalibrationEvaluatorSettings,
        procedure_type: str,
        model_ckpt_freq: int,
        update_validation_set: bool = False,
    ):
        self.num_iter = num_iter
        self.label_k = label_k
        self.init_num_labeled = init_num_labeled
        self.procedure_type = procedure_type
        self.settings = settings
        self.settings.device = torch.device("cpu")
        self.model_ckpt_freq = model_ckpt_freq
        self.update_validation_set = update_validation_set

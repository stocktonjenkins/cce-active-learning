from pathlib import Path

import torch
from dataclasses import dataclass

from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.utils.generic_utils import get_yaml

@dataclass
class DeteministicSettings:
    seed: int = 0
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    disable_debug_apis: bool = True

class ActiveLearningConfig:
    @staticmethod
    def from_yaml(config_path: str | Path) -> "ActiveLearningConfig":
        config_dict = get_yaml(config_path)
        config_dict["settings"] = CalibrationEvaluatorSettings(
            **{**config_dict["settings"], "device": torch.device("cpu")}
        )
        config_dict["deterministic_settings"] = DeteministicSettings(
            **{**config_dict["deterministic_settings"]}
        )
        return ActiveLearningConfig(**config_dict)

    def __init__(
        self,
        deterministic_settings: DeteministicSettings,
        num_iter: int,
        label_k: int,
        init_num_labeled: int,
        settings: CalibrationEvaluatorSettings,
        procedure_type: str,
        model_ckpt_freq: int,
        update_validation_set: bool = False,
    ):
        self.deterministic_settings = deterministic_settings
        self.num_iter = num_iter
        self.label_k = label_k
        self.init_num_labeled = init_num_labeled
        self.procedure_type = procedure_type
        self.settings = settings
        self.settings.device = torch.device("cpu")
        self.model_ckpt_freq = model_ckpt_freq
        self.update_validation_set = update_validation_set
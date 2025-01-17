from enum import Enum
from pathlib import Path

import torch
from dataclasses import dataclass

from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.utils.generic_utils import get_yaml


@dataclass
class DeterministicSettings:
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    disable_debug_apis: bool = True


class ProcedureType(Enum):
    RANDOM = "random"
    CCE = "cce"
    WEIGHTED_CCE = "weighted_cce"
    SOFTMAX_CCE = "softmax_cce"
    BAIT = "bait"
    BADGE = "badge"
    REVERSE_CCE = "reverse_cce"
    CONFIDENCE = "confidence"
    CORESET = "coreset"
    DROPOUT = "dropout"
    LCMD = "lcmd"
    BATCHBALD = "batchbald"


class   CheckpointType(Enum):
    BEST_LOSS = "best_loss"
    BEST_MAE = "best_mae"


class ActiveLearningConfig:
    procedure_type: ProcedureType

    @staticmethod
    def from_yaml(config_path: str | Path) -> "ActiveLearningConfig":
        config_dict = get_yaml(config_path)
        config_dict["settings"] = CalibrationEvaluatorSettings(
            **{**config_dict["settings"], "device": torch.device("cpu")}
        )
        config_dict["deterministic_settings"] = DeterministicSettings(
            **{**config_dict["deterministic_settings"]}
        )
        return ActiveLearningConfig(**config_dict)

    def __init__(
        self,
        deterministic_settings: DeterministicSettings,
        seeds: list[int],
        num_al_iter: int,
        label_k: int,
        init_num_labeled: int,
        chkp_type: str,
        settings: CalibrationEvaluatorSettings,
        num_workers: int,
        measure_calibration: bool = False,
        update_validation_set: bool = False,
        wandb: bool = False,
    ):
        self.deterministic_settings = deterministic_settings
        self.seeds = seeds
        self.num_al_iter = num_al_iter
        self.label_k = label_k
        self.init_num_labeled = init_num_labeled
        self.chkp_type = CheckpointType._value2member_map_[chkp_type]
        self.settings = settings
        self.settings.device = torch.device("cpu")
        self.update_validation_set = update_validation_set
        self.measure_calibration = measure_calibration
        self.wandb = wandb
        self.num_workers = num_workers

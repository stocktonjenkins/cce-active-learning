from pathlib import Path

from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.utils.generic_utils import get_yaml


class ActiveLearningConfig:
    @staticmethod
    def from_yaml(config_path: str | Path) -> "ActiveLearningConfig":
        config_dict = get_yaml(config_path)
        config_dict['settings'] = CalibrationEvaluatorSettings(**config_dict['settings'])

        return ActiveLearningConfig(**config_dict)

    def __init__(
        self,
        num_iter: int,
        label_k: int,
        init_num_labeled: int,
        settings: CalibrationEvaluatorSettings,
        procedure_type: str,
    ):
        self.num_iter = num_iter
        self.label_k = label_k
        self.init_num_labeled = init_num_labeled
        self.procedure_type = procedure_type
        self.settings = CalibrationEvaluatorSettings

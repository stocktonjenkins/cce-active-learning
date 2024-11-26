from probcal.active_learning.configs import ActiveLearningConfig
from probcal.active_learning.procedures.cce_active_learning_procedure import (
    CCEProcedure,
)
from probcal.active_learning.procedures.random_procedure import RandomProcedure


def get_active_learning_procedure(config: ActiveLearningConfig):
    if config.procedure_type == "random":
        return RandomProcedure
    if config.procedure_type == "cce":
        return CCEProcedure
    else:
        raise ValueError(
            f"Unknown active learning procedure type: {config.procedure_type}"
        )

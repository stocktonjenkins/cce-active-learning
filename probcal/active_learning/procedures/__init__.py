from probcal.active_learning.configs import ActiveLearningConfig, ProcedureType
from probcal.active_learning.procedures.cce_active_learning_procedure import (
    CCEProcedure,
)
from probcal.active_learning.procedures.confidence_active_learning_procedure import (
    ConfidenceProcedure,
)
from probcal.active_learning.procedures.random_procedure import RandomProcedure
from probcal.active_learning.procedures.bait_procedure import (
    BAITProcedure,
)
from probcal.active_learning.procedures.badge_procedure import (
    BadgeProcedure,
)
from probcal.active_learning.procedures.dropout_procedure import (
    DropoutProcedure,
)
from probcal.active_learning.procedures.reverse_cce_active_learning_procedure import (
    ReverseCCEProcedure,
)
from probcal.active_learning.procedures.coreset_procedure import (
    CoreSetProcedure,
)
from probcal.active_learning.procedures.batchbald_procedure import (
    BatchBALDProcedure,
)


def get_active_learning_procedure(config: ActiveLearningConfig):
    if config.procedure_type == ProcedureType.RANDOM:
        return RandomProcedure
    if config.procedure_type == ProcedureType.CCE:
        return CCEProcedure
    if config.procedure_type == ProcedureType.BAIT:
        return BAITProcedure
    if config.procedure_type == ProcedureType.CORESET:
        return CoreSetProcedure
    if config.procedure_type == ProcedureType.BADGE:
        return BadgeProcedure
    if config.procedure_type == ProcedureType.REVERSE_CCE:
        return ReverseCCEProcedure
    if config.procedure_type == ProcedureType.CONFIDENCE:
        return ConfidenceProcedure
    if config.procedure_type == ProcedureType.DROPOUT:
        return DropoutProcedure
    if config.procedure_type == ProcedureType.BATCHBALD:
        return BatchBALDProcedure
    else:
        raise ValueError(
            f"Unknown active learning procedure type: {config.procedure_type}"
        )

import random

import numpy as np
import torch
from torch import distributed

from probcal.active_learning.configs import DeterministicSettings


def seed_torch(seed: int, settings: DeterministicSettings):
    """Setup random seeds and other torch details"""
    if settings.disable_debug_apis:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if settings.cudnn_deterministic:
        torch.backends.cudnn.deterministic = settings.cudnn_deterministic
    if settings.cudnn_benchmark:
        torch.backends.cudnn.benchmark = settings.cudnn_benchmark


class SingleProcessContext:
    def __enter__(self):
        """
        Checks if the current process is the first process of distributed training.
        """
        return not distributed.is_initialized() or distributed.get_rank() == 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Synchronizes all processes.
        """
        distributed.barrier()

import os
import random

import numpy as np
import torch

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


def run_single_process():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if not cuda_visible_devices:
        return torch.cuda.current_device() == 0
    cuda_visible_devices = list(map(int, cuda_visible_devices.split(",")))
    return (
        min(cuda_visible_devices) == torch.cuda.current_device()
        or len(cuda_visible_devices) == 1
    )

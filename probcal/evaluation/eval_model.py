import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Type

import lightning as L
import torch
import yaml

from probcal.enums import DatasetType
from probcal.evaluation.calibration_evaluator import CalibrationEvaluator
from probcal.evaluation.calibration_evaluator import CalibrationEvaluatorSettings
from probcal.evaluation.kernels import rbf_kernel
from probcal.models.discrete_regression_nn import DiscreteRegressionNN
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


def main(config_path: Path):

    config = EvaluationConfig.from_yaml(config_path)
    if not config.log_dir.exists():
        os.makedirs(config.log_dir)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
    )

    initializer: Type[DiscreteRegressionNN] = get_model(config, return_initializer=True)[1]
    model = initializer.load_from_checkpoint(config.model_ckpt_path)
    evaluator = L.Trainer(
        accelerator=config.accelerator_type.value,
        enable_model_summary=False,
        logger=False,
        devices=1,
        num_nodes=1,
    )
    metrics: dict = evaluator.test(model=model, datamodule=datamodule)[0]
    metrics = {k: float(v) for k, v in metrics.items()}

    if config.dataset_type == DatasetType.TABULAR and config.input_dim == 1:
        x_vals = torch.cat([x for x, _ in datamodule.test_dataloader()], dim=0)
        gamma = (1 / (2 * x_vals.var())).item()
        mcmd_input_kernel = partial(rbf_kernel, gamma=gamma)
    else:
        mcmd_input_kernel = "polynomial"

    calibration_eval_settings = CalibrationEvaluatorSettings(
        dataset_type=config.dataset_type,
        device=torch.device(config.accelerator_type.value),
        mcmd_num_trials=config.mcmd_num_trials,
        mcmd_input_kernel=mcmd_input_kernel,
        mcmd_output_kernel=config.mcmd_output_kernel,
        mcmd_lambda=config.mcmd_lambda,
        mcmd_num_samples=config.mcmd_num_samples,
        ece_bins=config.ece_bins,
        ece_weights=config.ece_weights,
        ece_alpha=config.ece_alpha,
    )
    calibration_evaluator = CalibrationEvaluator(settings=calibration_eval_settings)
    results = calibration_evaluator(model=model, data_module=datamodule)

    metrics.update(
        mean_mcmd=[float(result.mean_mcmd) for result in results.mcmd_results],
        ece=float(results.ece),
    )
    with open(config.log_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f)
    results.save(config.log_dir / "calibration_results.npz")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to evaluation config.yaml.")
    args = parser.parse_args()
    main(config_path=Path(args.config))

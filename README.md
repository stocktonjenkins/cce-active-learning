# Quantifying the Calibration of Probabilistic Regression Models

This repository contains the official implementation of "A General Method for Measuring Calibration of Probabilistic Neural Regressors".

## Important Links

Important figures used in the paper, along with the code that generated them, can be found in [this directory](probcal/figures).

Our implementations of the probabilistic neural networks referenced in the paper can be found at the following locations:

- [Gaussian DNN](probcal/models/gaussian_nn.py)
- [Poisson DNN](probcal/models/poisson_nn.py)
- [NB DNN](probcal/models/neg_binom_nn.py)

Saved model weights can be found [here](weights), and dataset files can be found [here](data). Configs to reproduce the models referenced in the paper are saved in the [configs](configs) directory.

## Install Project Dependencies

```bash
conda create --name probcal python=3.10
conda activate probcal
pip install -r requirements.txt
```

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting. There will also be a flake8 check that provides warnings about various Python styling violations. These must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

## Training models

To train a probabilistic neural network, first fill out a config (using [this config](probcal/training/sample_train_config.yaml) as a template). Then, from the terminal, run

```bash
python probcal/training/train_model.py --config path/to/your/config.yaml
```

Logs / saved model weights will be found at the locations specified in your config.

### Training on Tabular Datasets

If fitting a model on tabular data, the training script assumes the dataset will be stored locally in `.npz` files with `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test` splits. Pass a path to this `.npz` file in the `dataset` `path` key in the config (also ensure that the `dataset` `type` is set to `tabular` and the `dataset` `input_dim` key is properly specified).

### Adding New Models

All regression models should inherit from the `DiscreteRegressionNN` class (found [here](probcal/models/discrete_regression_nn.py)). This base class is a `lightning` module, which allows for a lot of typical NN boilerplate code to be abstracted away. Beyond setting a few class attributes like `loss_fn` while calling the super-initializer, the only methods you need to actually write to make a new module are:

- `_forward_impl` (defines a forward pass through the network)
- `_predict_impl` (defines how to make predictions with the network, including any transformations on the output of the forward pass)
- `_sample_impl` (defines how to sample from the network's learned posterior predictive distribution for a given input)
- `_posterior_predictive_impl` (defines how to produce a posterior predictive distribution from network output)
- `_point_prediction_impl` (defines how to interpret network output as a single point prediction for a regression target)
- `_addl_test_metrics_dict` (defines any metrics beyond rmse/mae that are computed during model evaluation)
- `_update_addl_test_metrics_batch` (defines how to update additional metrics beyond rmse/mae for each test batch).

See existing model classes like `GaussianNN` (found [here](probcal/models/gaussian_nn.py)) for an example of these steps.

## Evaluating Models

To obtain evaluation metrics for a given model, first fill out a config (using [this config](probcal/evaluation/sample_eval_config.yaml) as a template).
Then, run the following command:

```bash
python probcal/evaluation/eval_model.py --config path/to/eval/config.yaml
```

Two results files will be saved to the `log_dir` you specify in your config:

- A `test_metrics.yaml` with metrics like MAE, RMSE, etc. and a summary of the calibration results (such as the mean MCMDs for each specified trial)
- A `calibration_results.npz` file which can be loaded into a `CalibrationResults` object to see granular MCMD and ECE results.

## Measuring Calibration

Once a `DiscreteRegressionNN` subclass is trained, its calibration can be measured on a dataset via the `CalibrationEvaluator`. Example usage:

```python
from probcal.data_modules import COCOPeopleDataModule
from probcal.enums import DatasetType
from probcal.evaluation import CalibrationEvaluator
from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.models import GaussianNN


# You can customize the settings for the MCMD / ECE computation.
settings = CalibrationEvaluatorSettings(
    dataset_type=DatasetType.IMAGE,
    mcmd_input_kernel="polynomial",
    mcmd_output_kernel="rbf",
    mcmd_lambda=0.1,
    mcmd_num_samples=5,
    ece_bins=50,
    ece_weights="frequency",
    ece_alpha=1,
    # etc.
)
evaluator = CalibrationEvaluator(settings)

model = GaussianNN.load_from_checkpoint("path/to/model.ckpt")

# You can use any lightning data module (preferably, the one with the dataset the model was trained on).
data_module = COCOPeopleDataModule(root_dir="data", batch_size=4, num_workers=0, persistent_workers=False)
calibration_results = evaluator(model=model, data_module=data_module)
calibration_results.save("path/to/results.npz")
```

Invoking the `CalibrationEvaluator`'s `__call__` method (as above) kicks off an extensive evaluation wherein MCMD and ECE are computed for the specified model. This passes back a `CalibrationResults` object, which will contain the computed metrics and other helpful variables for further analysis.

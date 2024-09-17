from __future__ import annotations

from pathlib import Path

import yaml

from probcal.enums import AcceleratorType
from probcal.enums import BetaSchedulerType
from probcal.enums import DatasetType
from probcal.enums import HeadType
from probcal.enums import ImageDatasetName
from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.utils.generic_utils import get_yaml
from probcal.utils.generic_utils import to_snake_case


class BaseConfig(object):
    def __init__(
        self,
        experiment_name: str,
        head_type: HeadType,
        dataset_type: DatasetType,
        dataset_path_or_spec: Path | ImageDatasetName,
        source_dict: dict,
        hidden_dim: int,
    ):
        self.experiment_name = experiment_name
        self.head_type = head_type
        self.dataset_type = dataset_type
        self.dataset_path_or_spec = dataset_path_or_spec
        self.source_dict = source_dict
        self.hidden_dim = hidden_dim

    def to_yaml(self, filepath: str | Path):
        """Save this config as a .yaml file at the given filepath.

        Args:
            filepath (str | Path): The filepath to save this config at.
        """
        with open(filepath, "w") as f:
            yaml.dump(self.source_dict, f)

    @staticmethod
    def from_yaml(config_path: str | Path) -> TrainingConfig:
        """Factory method to construct an TrainingConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            TrainingConfig: The specified config.
        """
        raise NotImplementedError("This method should be implemented in a subclass of BaseConfig.")

    @staticmethod
    def get_dataset_path_or_spec(dataset_cfg: dict):
        if "path" in dataset_cfg:
            return Path(dataset_cfg["path"])
        else:
            type = DatasetType(dataset_cfg["type"])
            spec = dataset_cfg["spec"]
            if type == DatasetType.TABULAR:
                return Path(dataset_cfg["path"])
            if type == DatasetType.IMAGE:
                return ImageDatasetName(spec)
            else:
                return Path(spec)


class TrainingConfig(BaseConfig):
    """Class with configuration options for training a model.

    Attributes:
        experiment_name (str): The name of the training run (used for identifying chkp weights / eval logs), automatically cast to snake case.
        head_type (HeadType): The output head to use in the neural network, e.g. "gaussian", "poisson", etc.
        chkp_dir (Path): Directory to checkpoint model weights in.
        chkp_freq (int): Number of epochs to wait in between checkpointing model weights.
        batch_size (int): The batch size to train with.
        num_epochs (int): The number of epochs through the data to complete during training.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        dataset_type (DatasetType): Type of dataset to use in this experiment.
        dataset_path_or_spec (Path | ImageDatasetName): If specifying a tabular dataset, path to the dataset .npz file to use. Otherwise, the name of the dataset.
        num_trials (int): Number of trials to run for this experiment.
        log_dir (Path): Directory to log results to.
        source_dict (dict): Dictionary from which config was constructed.
        input_dim (int): The input dim of the data (used to construct the MLP). Defaults to 1.
        hidden_dim (int, optional): Feature dimension used in the model (before feeding the representation to the output head). Defaults to 64.
        precision (str | None, optional): String specifying desired floating point precision for training. Defaults to None.
        random_seed (int | None, optional): If specified, the random seed to use for reproducibility. Defaults to None.
    """

    def __init__(
        self,
        experiment_name: str,
        accelerator_type: AcceleratorType,
        head_type: HeadType,
        chkp_dir: Path,
        chkp_freq: int,
        batch_size: int,
        num_epochs: int,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None,
        lr_scheduler_kwargs: dict | None,
        beta_scheduler_type: BetaSchedulerType | None,
        beta_scheduler_kwargs: dict | None,
        dataset_type: DatasetType,
        dataset_path_or_spec: Path | ImageDatasetName,
        num_trials: int,
        log_dir: Path,
        source_dict: dict,
        input_dim: int = 1,
        hidden_dim: int = 64,
        precision: str | None = None,
        random_seed: int | None = None,
    ):
        super(TrainingConfig, self).__init__(
            experiment_name=experiment_name,
            head_type=head_type,
            dataset_type=dataset_type,
            source_dict=source_dict,
            dataset_path_or_spec=dataset_path_or_spec,
            hidden_dim=hidden_dim,
        )
        self.accelerator_type = accelerator_type
        self.chkp_dir = chkp_dir
        self.chkp_freq = chkp_freq
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.beta_scheduler_type = beta_scheduler_type
        self.beta_scheduler_kwargs = beta_scheduler_kwargs
        self.num_trials = num_trials
        self.log_dir = log_dir
        self.input_dim = input_dim
        self.precision = precision
        self.random_seed = random_seed

    @staticmethod
    def from_yaml(config_path: str | Path) -> TrainingConfig:
        """Factory method to construct an TrainingConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            TrainingConfig: The specified config.
        """
        config_dict = get_yaml(config_path)
        training_dict: dict = config_dict["training"]
        eval_dict: dict = config_dict["evaluation"]

        experiment_name = to_snake_case(config_dict["experiment_name"])
        accelerator_type = AcceleratorType(training_dict["accelerator"])
        head_type = HeadType(config_dict["head_type"])
        chkp_dir = Path(training_dict["chkp_dir"])
        chkp_freq = training_dict["chkp_freq"]
        batch_size = training_dict["batch_size"]
        num_epochs = training_dict["num_epochs"]
        precision = training_dict.get("precision")
        optim_type = OptimizerType(training_dict["optimizer"]["type"])
        optim_kwargs = training_dict["optimizer"]["kwargs"]

        if "lr_scheduler" in training_dict:
            lr_scheduler_type = LRSchedulerType(training_dict["lr_scheduler"]["type"])
            lr_scheduler_kwargs = training_dict["lr_scheduler"]["kwargs"]
        else:
            lr_scheduler_type = None
            lr_scheduler_kwargs = None

        if "beta_scheduler" in training_dict:
            beta_scheduler_type = BetaSchedulerType(training_dict["beta_scheduler"]["type"])
            beta_scheduler_kwargs = training_dict["beta_scheduler"]["kwargs"]
            if beta_scheduler_kwargs.get("last_epoch", None) == -1:
                beta_scheduler_kwargs["last_epoch"] = num_epochs
        else:
            beta_scheduler_type = None
            beta_scheduler_kwargs = None

        dataset_type = DatasetType(config_dict["dataset"]["type"])
        dataset_path_or_spec = TrainingConfig.get_dataset_path_or_spec(config_dict["dataset"])

        num_trials = eval_dict["num_trials"]
        log_dir = Path(eval_dict["log_dir"])
        input_dim = config_dict["dataset"].get("input_dim", 1)
        hidden_dim = config_dict.get("hidden_dim", 64)
        random_seed = config_dict.get("random_seed")

        return TrainingConfig(
            experiment_name=experiment_name,
            accelerator_type=accelerator_type,
            head_type=head_type,
            chkp_dir=chkp_dir,
            chkp_freq=chkp_freq,
            batch_size=batch_size,
            num_epochs=num_epochs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            beta_scheduler_type=beta_scheduler_type,
            beta_scheduler_kwargs=beta_scheduler_kwargs,
            dataset_type=dataset_type,
            dataset_path_or_spec=dataset_path_or_spec,
            num_trials=num_trials,
            log_dir=log_dir,
            source_dict=config_dict,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            precision=precision,
            random_seed=random_seed,
        )


class TestConfig(BaseConfig):
    """Class with configuration options for testing a model"""

    def __init__(
        self,
        experiment_name: str,
        head_type: HeadType,
        dataset_type: DatasetType,
        source_dict: dict,
        dataset_path_or_spec: Path | ImageDatasetName,
        hidden_dim,
    ):
        super(TestConfig, self).__init__(
            experiment_name=experiment_name,
            head_type=head_type,
            dataset_type=dataset_type,
            source_dict=source_dict,
            dataset_path_or_spec=dataset_path_or_spec,
            hidden_dim=hidden_dim,
        )

    @staticmethod
    def from_yaml(config_path: str | Path) -> TestConfig:
        """Factory method to construct an TestConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            TestConfig: The specified config.
        """
        config_dict = get_yaml(config_path)
        experiment_name = to_snake_case(config_dict["experiment_name"])
        head_type = HeadType(config_dict["head_type"])
        dataset_type = DatasetType(config_dict["dataset"]["type"])
        hidden_dim = config_dict.get("hidden_dim", 64)
        dataset_path_or_spec = TrainingConfig.get_dataset_path_or_spec(config_dict["dataset"])

        return TestConfig(
            experiment_name=experiment_name,
            head_type=head_type,
            dataset_type=dataset_type,
            source_dict=config_dict,
            dataset_path_or_spec=dataset_path_or_spec,
            hidden_dim=hidden_dim,
        )

import torch
from lightning import Callback, LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


class WandBLoggingCallback(Callback):
    def __init__(self, exp_name: str, logger: WandbLogger):
        super(WandBLoggingCallback, self).__init__()
        self.exp_name = exp_name
        self.logger = logger

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = {f"{self.exp_name}/epoch": trainer.current_epoch}
        for key, value in trainer.callback_metrics.items():
            metrics[f"{self.exp_name}/{key}"] = (
                value.item() if isinstance(value, torch.Tensor) else value
            )
        self.logger.log_metrics(metrics, step=trainer.current_epoch)


class WandBActiveLearningCallback(WandBLoggingCallback):
    def __init__(self, logger: WandbLogger, al_iter: int):
        super(WandBActiveLearningCallback, self).__init__(
            exp_name=f"AL{al_iter}", logger=logger
        )

import math


class HyperparamScheduler:
    """Base class for all parameter schedulers (used to update the value of schedulable hyperparameters during training)."""

    def __init__(self, x_0: float = 0.0, x_1: float = 0.0, num_steps: int = 1000):
        """Initialize a HyperparamScheduler.

        Args:
            x_0 (float, optional): Initial parameter value. Defaults to 0.0.
            x_1 (float, optional): Final parameter value. Defaults to 0.0.
            num_steps (int, optional): Number of steps to form schedule for. Defaults to 1000.
        """

        self.x_0 = x_0
        self.x_1 = x_1
        self.num_steps = num_steps

        self.current_value = x_0
        self.current_step = 0

    def step(self):
        """Update internal counter for the current step, and the current value of this scheduler."""
        self.current_step += 1
        self._update_current_value()

    def _update_current_value(self):
        """Update `self.current_value` according to the specified schedule."""
        raise NotImplementedError("Should be implemented by subclass.")


class CosineAnnealingScheduler(HyperparamScheduler):
    """A scheduler that uses cosine annealing to gradually change from `x_0` to `x_1`."""

    def _update_current_value(self):
        self.current_value = self.x_1 + 0.5 * (self.x_0 - self.x_1) * (
            1 + math.cos((self.current_step * math.pi) / self.num_steps)
        )


class LinearScheduler(HyperparamScheduler):
    """A scheduler that uses linear steps to gradually change from `x_0` to `x_1`."""

    def _update_current_value(self):
        self.current_value = self.x_0 - self.current_step * (
            (self.x_0 - self.x_1) / self.num_steps
        )

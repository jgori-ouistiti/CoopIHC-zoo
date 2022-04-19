import numpy as np


class ExponentialDecayMemory:
    @staticmethod
    def decay(
        delta_time=None,
        times_presented=None,
        initial_forgetting_rate=None,
        repetition_effect=None,
        log=False,
    ):

        forget_rate = initial_forgetting_rate * (1 - repetition_effect) ** (
            times_presented
        )

        if log:
            return -forget_rate * delta_time
        else:
            with np.errstate(divide="ignore", over="ignore"):  # invalid="ignore",
                return np.exp(-forget_rate * delta_time)

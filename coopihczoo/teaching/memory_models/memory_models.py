import numpy as np


EPS = np.finfo(np.float).eps


class ExponentialDecayMemory:
    @staticmethod
    def decay(
        delta_time,
        times_presented,
        initial_forgetting_rate,
        repetition_effect,
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
    
    @staticmethod
    def log_like_grid(
            times_presented,
            delta_time,
            grid_param,
            item,
            response):
        
        fr = grid_param[:, 0] \
             * (1 - grid_param[:, 1]) ** (times_presented[item] - 1)

        delta = delta_time[item]

        # delta *= cst_time
        p_success = np.exp(- fr * delta)

        p = p_success if response else 1-p_success

        log_lik = np.log(p + EPS)
        return log_lik

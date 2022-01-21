from coopihczoo.eye.envs import ChenEyePointingTask
from coopihczoo.eye.users import ChenEye
from coopihczoo.eye.utils import eccentric_noise

from coopihc.bundle.Bundle import Bundle
from coopihc.space.State import State

import numpy
import matplotlib.pyplot as plt


fitts_W = 4e-2
fitts_D = 0.8
perceptualnoise = 0.1
oculomotornoise = 0.1
task = ChenEyePointingTask(fitts_W, fitts_D, threshold=0.05, dimension=2)
user = ChenEye(perceptualnoise, oculomotornoise, dimension=2)
bundle = Bundle(task=task, user=user)
game_state = bundle.reset(turn=1)
bundle.render("plotext")


def eccentric_motor_noise(action, observation, oculomotornoise, *args, **kwargs):

    noise_covariance_matrix = eccentric_noise(
        action, observation["task_state"]["fixation"], oculomotornoise
    )

    noise = numpy.random.multivariate_normal(
        numpy.zeros(shape=action.reshape(-1).shape), noise_covariance_matrix
    )
    return noise, noise_covariance_matrix


while True:
    action = bundle.user.observation["user_state"]["belief-mu"]
    noise = eccentric_motor_noise(action, bundle.user.observation, oculomotornoise)[0]
    noisy_action = action.reshape(-1, 1) + noise.reshape(-1, 1)

    obs, rewards, is_done = bundle.step(noisy_action)
    bundle.render("plotext")

    if is_done:
        plt.pause(1)
        break

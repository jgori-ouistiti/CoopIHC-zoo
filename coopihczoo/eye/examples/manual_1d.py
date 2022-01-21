from coopihc.bundle.Bundle import Bundle
from coopihc.space.State import State

import matplotlib.pyplot as plt
import numpy

from coopihczoo.eye.envs import ChenEyePointingTask
from coopihczoo.eye.users import ChenEye
from coopihczoo.eye.utils import eccentric_noise

fitts_W = 4e-2
fitts_D = 0.8
perceptualnoise = 0.09
oculomotornoise = 0.09
task = ChenEyePointingTask(fitts_W, fitts_D, dimension=1)
user = ChenEye(perceptualnoise, oculomotornoise, dimension=1)
bundle = Bundle(task=task, user=user)
bundle.reset(turn=1)
bundle.render("plot")


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
    noisy_action = action + noise
    obs, reward_list, is_done = bundle.step(user_action=noisy_action)
    # bundle.render("plotext")
    bundle.render("plot")

    if is_done:
        # plt.pause(100)
        bundle.close()
        break

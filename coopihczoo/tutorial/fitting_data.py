from coopihczoo.eye.envs import ChenEyePointingTask
from coopihczoo.eye.users import ChenEye

from coopihc.bundle.Bundle import Bundle

import numpy
import scipy.optimize as opti
import matplotlib.pyplot as plt


def create_conditions(D, motor_noise_std):
    return {
        "task": ChenEyePointingTask,
        "task_args": (target_size, D),
        "user": ChenEye,
        "user_args": (perceptual_noise_std, motor_noise_std),
    }


def create_and_sample_bundle(motor_noise_std, D):
    bundle_dict = create_conditions(D, motor_noise_std)
    return Bundle(
        task=bundle_dict["task"](*bundle_dict["task_args"]),
        user=bundle_dict["user"](*bundle_dict["user_args"]),
    )


def evaluate_saccade_number_per_id(motor_noise_std, IDlvls, n=100):
    """For a given motor noise level, get the mean and standard deviations of the number of saccades for various levels of ID"""
    container = []
    for _id in IDlvls:
        D = target_size * 2 ** (_id - 1)
        bundle = create_and_sample_bundle(motor_noise_std, D)
        data = bundle.sample(n_turns=n)
        _mean, _std = numpy.mean([len(i[0]) for i in data[0]]), numpy.std(
            [len(i[0]) for i in data[0]]
        )
        container.append((_id, _mean, _std))
    return container


def cost(container):
    """MSE between the average number of saccades from the simulation and the empirical results in Fig.4 https://dl.acm.org/doi/fullHtml/10.1145/3290605.3300765"""
    _sum = 0
    for _id, _mean, _ in container:
        _sum += ((1 + (_id - 2.0) / 3.0) - _mean) ** 2

    return _sum


def objective(motor_noise_std):
    """Objective function to minimize"""
    IDlvls = [1, 2, 3, 4, 5]
    container = evaluate_saccade_number_per_id(motor_noise_std, IDlvls)
    return cost(container)


target_size = 4e-2
target_distance = 0.8
perceptual_noise_std = 0.09

# simple grid search with 10 steps
results = []
for motor_noise in numpy.linspace(0.01, 0.1, 10):
    results.append((motor_noise, objective(motor_noise)))

motor_noise = sorted(results, key=lambda x: x[1])[0][0]

motor_noise_std = motor_noise

# simulate resulting behavior

task = ChenEyePointingTask(target_size, target_distance)
user = ChenEye(perceptual_noise_std, motor_noise_std)
bundle = Bundle(task=task, user=user)

# Plot resulting behavior
# Interaction loop starts
bundle.reset()
bundle.render("plottext")
plt.tight_layout()

while True:
    obs, rewards, is_done = bundle.step()
    bundle.render("plottext")
    if is_done:
        # bundle.close()
        break
# Interaction loop ends

from coopihczoo.eye.envs import ChenEyePointingTask
from coopihczoo.eye.users import ChenEye

from coopihc.bundle.Bundle import Bundle

import matplotlib.pyplot as plt


target_size = 4e-2
target_distance = 0.8
motor_noise_std = 0.09
perceptual_noise_std = 0.09
task = ChenEyePointingTask(target_size, target_distance)
user = ChenEye(perceptual_noise_std, motor_noise_std)
bundle = Bundle(task=task, user=user)

# Interaction loop starts
bundle.reset()
bundle.render("plottext")
plt.tight_layout()

while True:
    obs, rewards, is_done = bundle.step()
    bundle.render("plottext")
    if is_done:
        bundle.close()
        break
# Interaction loop ends

# Interaction loop is equivalent to this (without render)
data = bundle.sample(n_turns=1)

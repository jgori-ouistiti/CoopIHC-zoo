from coopihczoo.eye.envs import ChenEyePointingTask
from coopihczoo.eye.users import ChenEye
from coopihczoo.eye.utils import eccentric_noise

from coopihc.bundle.Bundle import Bundle
from coopihc.space.State import State

import numpy
import matplotlib.pyplot as plt


fitts_W = 4e-2
fitts_D = 0.8
ocular_std = 0.09
swapping_std = 0.09
task = ChenEyePointingTask(fitts_W, fitts_D)
user = ChenEye(swapping_std, ocular_std)
bundle = Bundle(task=task, user=user)
bundle.reset()
bundle.render("plot")
while True:
    obs, rewards, is_done = bundle.step()
    bundle.render("plot")
    if is_done:
        break

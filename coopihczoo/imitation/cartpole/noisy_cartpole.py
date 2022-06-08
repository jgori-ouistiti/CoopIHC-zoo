"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Modified by Aaditya Ravindran to include friction and random sensor & actuator noise
"""

import math
import warnings

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class CartPoleModEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, case=1):
        self.__version__ = "0.2.0"
        print(
            "CartPoleModEnv - Version {}, Noise case: {}".format(self.__version__, case)
        )
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self._seed()
        if case < 4:
            self.force_mag = 10.0 * (1 + self.addnoise(case))
            self.case = 1
        else:
            self.force_mag = 10.0
            self.case = case

        self.tau = 0.02  # seconds between state updates
        self.frictioncart = 5e-4  # AA Added cart friction
        self.frictionpole = 2e-6  # AA Added cart friction

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )

        self.action_space = spaces.Discrete(2)  # AA Set discrete states back to 2
        self.observation_space = spaces.Box(-high, high)

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, *args, **kwargs):
        return self._seed(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._close(*args, **kwargs)

    def addnoise(self, x):
        return np.array(0).reshape((1,))
        # return {
        #     1: np.array(0).reshape((1,)),
        #     2: self.np_random.uniform(
        #         low=-0.05, high=0.05, size=(1,)
        #     ),  #  5% actuator noise
        #     3: self.np_random.uniform(
        #         low=-0.10, high=0.10, size=(1,)
        #     ),  # 10% actuator noise
        #     4: self.np_random.uniform(
        #         low=-0.05, high=0.05, size=(1,)
        #     ),  #  5% sensor noise
        #     5: self.np_random.uniform(
        #         low=-0.10, high=0.10, size=(1,)
        #     ),  # 10% sensor noise
        #     6: self.np_random.normal(
        #         loc=0, scale=np.sqrt(0.10), size=(1,)
        #     ),  # 0.1 var sensor noise
        #     7: self.np_random.normal(
        #         loc=0, scale=np.sqrt(0.20), size=(1,)
        #     ),  # 0.2 var sensor noise
        # }.get(x, 1)

    def _seed(self, seed=None):  # Set appropriate seed value
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force
            + self.polemass_length * theta_dot * theta_dot * sintheta
            - self.frictioncart * np.sign(x_dot)
        ) / self.total_mass  # AA Added cart friction
        thetaacc = (
            self.gravity * sintheta
            - costheta * temp
            - self.frictionpole * theta_dot / self.polemass_length
        ) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )  # AA Added pole friction
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        noise = self.addnoise(self.case)
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = (theta + self.tau * theta_dot) * (1 + noise.squeeze().tolist())
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = [x, float(x_dot), theta, float(theta_dot)]
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                warnings.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)


from gym.envs.registration import register

for i in range(7):
    register(
        id=f"CartPole-v{i-1}",
        entry_point="gym.envs.classic_control:CartPoleEnv",
        max_episode_steps=200,
        reward_threshold=195.0,
        kwargs={"case": i},
    )

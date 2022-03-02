from coopihc import Bundle, TrainGym

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.rl import Teacher

from gym.wrappers import FilterObservation
import gym

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

import os


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = \
            gym.spaces.Discrete(env.action_space["assistant_action"].n)

    def action(self, action):
        return {"assistant_action": int(action)}

    def reverse_action(self, action):
        return action["assistant_action"]


def run_rl():

    n_item = 5
    inter_trial = 1
    max_iter = 40
    is_item_specific = False
    param = 0.01, 0.2
    thr = 0.9

    task = Task(
        thr=thr,
        inter_trial=inter_trial,
        n_item=n_item,
        max_iter=max_iter,
        is_item_specific=is_item_specific,
    )
    user = User(param=param)
    assistant = Teacher(thr=thr)
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False)

    env = TrainGym(
        bundle,
        train_user=False,
        train_assistant=True,
    )
    _ = env.reset()

    # Dict(turn_index:Discrete(4), round_index:Discrete(1000), position:Discrete(31), targets:MultiDiscrete([31 31 31 31 31 31 31 31]), goal:Discrete(31), user_action:Discrete(3), assistant_action:Box(1.0, 1.0, (1, 1), float32))
    env.step({"assistant_action": 1})

    # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    from stable_baselines3.common.env_checker import check_env

    check_env(env, warn=False)

    # print(env.observation_space)

    env = FilterObservation(
        env,
        ("memory", "progress"))

    env = ActionWrapper(env)

    os.makedirs("tmp", exist_ok=True)

    env = Monitor(env, filename="tmp/log")

    model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./tb/")

    model.learn(total_timesteps=int(1e6))
    model.save("saved_model")


if __name__ == "__main__":
    run_rl()

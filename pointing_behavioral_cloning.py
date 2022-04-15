import torch
import numpy as np
import copy

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import os

from gym.wrappers import FilterObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from coopihc import Bundle, TrainGym

# from coopihczoo.teaching.users import User
# from coopihczoo.teaching.envs import Task
# from coopihczoo.teaching.assistants.rl import Teacher
# from coopihczoo.teaching.config import config_example
# from coopihczoo.teaching.assistants.conservative_sampling_expert import ConservativeSamplingExpert
from coopihczoo.teaching.rl.behavioral_cloning import BC

from coopihc.examples.simplepointing.users import CarefulPointer
from coopihc.examples.simplepointing.envs import SimplePointingTask
from coopihc.examples.simplepointing.assistants import BIGGain

from gym import ActionWrapper
from gym.spaces import Box


config_task = dict(gridsize=31, number_of_targets=8, mode="position")
config_user = dict(error_rate=0.05)

obs_keys = ("assistant_state__beliefs",
            "task_state__position", "task_state__targets",
            "user_action__action")


class AssistantActionWrapper(ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        _as = env.action_space["assistant_action__action"]
        self.action_space = Box(
            low=
            _as.low,
            high=_as.high,
            shape=_as.shape,
            dtype=_as.dtype
        )

    def action(self, action):
        return {"assistant_action__action": int(action)}

    def reverse_action(self, action):
        return action["assistant_action__action"]


def make_env():

    task = SimplePointingTask(**config_task)
    user = CarefulPointer(**config_user)
    assistant = BIGGain()
    bundle = Bundle(task=task, user=user, assistant=assistant,
                    random_reset=True,
                    reset_turn=3,
                    reset_skip_user_step=False)

    env = TrainGym(
        bundle,
        train_user=False,
        train_assistant=True)

    # # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    # check_env(env, warn=False)

    env = FilterObservation(
        env,
        obs_keys)

    env = AssistantActionWrapper(env)
    return env


def sample_expert():

    task = SimplePointingTask(**config_task)
    user = CarefulPointer(**config_user)
    assistant = BIGGain()

    bundle = Bundle(task=task, user=user, assistant=assistant)
    bundle.reset(go_to=3)
    # bundle.render("plotext")
    # plt.tight_layout()

    # plt.savefig("/home/juliengori/Pictures/img_tmp/biggain_{}.png".format(k))

    obs = assistant.observation.filter(mode="array-Gym", flat=True)
    obs_dic = {k: obs[k] for k in obs_keys}

    expert_data = [[], ]

    while True:
        state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
        # bundle.render("plotext")

        obs = assistant.observation.filter(mode="array-Gym", flat=True)
        new_obs_dic = {k: obs[k] for k in obs_keys}

        action = int(state.assistant_action["action"])

        expert_data[-1].append({"acts": action,     # .squeeze(),
                                "obs": obs_dic})    # .squeeze()})

        obs_dic = new_obs_dic

        # plt.savefig("/home/juliengori/Pictures/img_tmp/biggain_{}.png".format(k))
        if is_done:
            bundle.close()
            break

    np.random.shuffle(expert_data)

    # Flatten expert data
    flatten_expert_data = []
    for traj in expert_data:
        for e in traj:
            flatten_expert_data.append(e)

    expert_data = flatten_expert_data

    return expert_data

# ---- Main -------------


torch.manual_seed(1234)
np.random.seed(1234)

os.makedirs("tmp", exist_ok=True)

expert_data = sample_expert()

# n_env = 4
# envs = [make_env for _ in range(n_env)]
# vec_env = SubprocVecEnv(envs)
# vec_env = VecMonitor(vec_env, filename="tmp/log")

env = make_env()
env.reset()
exit()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tb/")
policy = model.policy

reward, _ = evaluate_policy(policy, Monitor(env), n_eval_episodes=10, render=False)
print(f"Reward before training: {reward}")

bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=expert_data,
    policy=policy)

print("Training a policy using Behavior Cloning")
bc_trainer.train()

reward, _ = evaluate_policy(model.policy, Monitor(env), n_eval_episodes=10, render=False)
print(f"Reward after training: {reward}")

# model.learn(total_timesteps=int(10e5), )
#
# reward, _ = evaluate_policy(model.policy, Monitor(env), n_eval_episodes=3, render=False)
# print(f"Reward after extended training: {reward}")


# def main():
#
#
#
#
# if __name__ == "__main__":
#     main()

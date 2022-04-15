import torch
import numpy as np

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import os

from gym.wrappers import FilterObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from coopihc import Bundle, TrainGym

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.config import config_example
from coopihczoo.teaching.assistants.conservative_sampling_expert import ConservativeSamplingExpert
from coopihczoo.teaching.action_wrapper.action_wrapper import AssistantActionWrapper

import torch

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from coopihczoo.teaching.rl.behavioral_cloning import BC
from coopihczoo.teaching.rl.dagger import DAgger


def make_env():

    task = Task(**config_example.task_kwargs)
    user = User(**config_example.user_kwargs)
    assistant = ConservativeSamplingExpert()
    bundle = Bundle(task=task, user=user, assistant=assistant,
                    random_reset=False,
                    reset_turn=3,
                    reset_skip_user_step=True)  # Begin by assistant

    env = TrainGym(
        bundle,
        train_user=False,
        train_assistant=True,
    )

    env = FilterObservation(
        env,
        ("memory", "progress"))

    env = AssistantActionWrapper(env)
    return env


def main():

    torch.manual_seed(1234)
    np.random.seed(1234)

    env = make_env()

    expert = env.unwrapped.bundle.assistant

    total_n_iter = \
        int(env.bundle.task.state["n_iter_per_ss"] * env.bundle.task.state["n_session"])

    model = A2C("MultiInputPolicy", Monitor(env), verbose=1, tensorboard_log="./tb/",
                # batch_size=total_n_iter,
                n_steps=total_n_iter)  # This is important to set for the learning to be effective!!

    policy = model.policy

    reward, _ = evaluate_policy(policy, Monitor(env), n_eval_episodes=10, render=False)
    print(f"Reward before training: {reward}")

    dagger_trainer = DAgger(
        env=env,
        expert=expert,
        policy=policy,
        batch_size=total_n_iter)

    dagger_trainer.train(2000)

    reward, _ = evaluate_policy(policy, Monitor(env), n_eval_episodes=10, render=False)
    print(f"Reward after training: {reward}")

    model.learn(total_timesteps=int(10e5), )

    reward, _ = evaluate_policy(model.policy, Monitor(env), n_eval_episodes=10, render=False)
    print(f"Reward after extended training: {reward}")


if __name__ == "__main__":
    main()


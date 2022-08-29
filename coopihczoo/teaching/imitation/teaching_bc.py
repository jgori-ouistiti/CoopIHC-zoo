import os

# import numpy as np

import pickle
from gym.wrappers import FilterObservation, FlattenObservation

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from gym import ActionWrapper
from gym.spaces import Discrete

from coopihc import Bundle, WrapperReferencer

from coopihczoo.teaching.users.users_naive_implementation import User
from coopihczoo.teaching.envs.envs_naive_implementation import Task
from coopihczoo.teaching.assistants.assistants_naive_implementation.conservative_sampling_expert import (
    ConservativeSamplingExpert,
)
from coopihczoo.teaching.config import config_example

from coopihczoo.utils.imitation.run import train_novice_behavioral_cloning_ppo
from coopihczoo.utils.imitation.behavioral_cloning import sample_expert


class AssistantActionWrapper(ActionWrapper, WrapperReferencer):

    def __init__(self, env):

        ActionWrapper.__init__(self, env)
        WrapperReferencer.__init__(self, env)

        self.action_space = Discrete(env.action_space["assistant_action__action"].high[0]+1)

    def action(self, action):
        return {"assistant_action__action": int(action)}

    def reverse_action(self, action):
        return action["assistant_action__action"]


def make_env(seed=123):

    task = Task(**config_example.task_kwargs)
    user = User(**config_example.user_kwargs)
    assistant = ConservativeSamplingExpert()
    bundle = Bundle(
        seed=seed,
        task=task,
        user=user,
        assistant=assistant,
        random_reset=False,
        reset_start_after=2,
        reset_go_to=3,
    )  # Begin by assistant

    obs_keys = ("assistant_state__memory", "assistant_state__progress")

    env = bundle.convert_to_gym_env(train_user=False, train_assistant=True)

    env = FlattenObservation(FilterObservation(env, obs_keys))
    env = AssistantActionWrapper(env)

    # Use env_checker from stable_baselines3 to verify that the make_env adheres to the Gym API
    check_env(env)
    return env


def main():

    evaluate_expert = True
    sample_expert_n_episode = 10000
    sample_expert_n_timestep = None

    bkp_folder = './tmp'
    samples_backup_file = f"{bkp_folder}/expert_samples.p"

    env = make_env()

    total_n_iter = int(
            env.bundle.task.state["n_iter_per_ss"]
            * env.bundle.task.state["n_session"]
        )

    novice_kwargs = dict(
        policy="MlpPolicy",
        batch_size=total_n_iter,
        n_steps=total_n_iter,
    )

    expert = env.unwrapped.assistant

    if not os.path.exists(samples_backup_file):

        print("Sampling expert...")

        expert_data = sample_expert(env=env, expert=expert,
                                    n_episode=sample_expert_n_episode,
                                    n_timestep=sample_expert_n_timestep,
                                    deterministic=True)

        with open(samples_backup_file, 'wb') as f:
            pickle.dump(expert_data, f)

    else:
        with open(samples_backup_file, 'rb') as f:
            expert_data = pickle.load(f)

    if evaluate_expert:
        print("Evaluating expert...")
        reward, _ = evaluate_policy(expert, Monitor(env), n_eval_episodes=50)
        print(f"Reward expert: {reward}")

    _ = train_novice_behavioral_cloning_ppo(
        make_env=make_env,
        novice_kwargs=novice_kwargs,
        expert_data=expert_data)


if __name__ == "__main__":
    main()

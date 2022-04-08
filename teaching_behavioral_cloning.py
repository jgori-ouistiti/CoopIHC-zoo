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
from coopihczoo.teaching.assistants.rl import Teacher
from coopihczoo.teaching.config import config_example
from coopihczoo.teaching.assistants.conservative_sampling_expert import ConservativeSamplingExpert
from coopihczoo.teaching.action_wrapper.action_wrapper import AssistantActionWrapper
from coopihczoo.teaching.rl.behavioral_cloning import BC


from teaching_rl_example import make_env


def sample_expert():

    task = Task(**config_example.task_kwargs)
    user = User(**config_example.user_kwargs)
    assistant = ConservativeSamplingExpert()
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False,
                    reset_start_after=2,
                    reset_go_to=3)
    bundle.reset()

    obs = assistant.state

    expert_data = [[], ]

    while True:

        state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
        new_obs = state["assistant_state"]
        action = int(state['assistant_action']["action"])

        obs_dic = {"memory": obs["memory"].view(np.ndarray),
                   "progress": obs["progress"].view(np.ndarray)}

        expert_data[-1].append({"acts": action,     # .squeeze(),
                                "obs": obs_dic})    # .squeeze()})

        obs = new_obs
        if is_done:
            break

    np.random.shuffle(expert_data)

    # Flatten expert data
    flatten_expert_data = []
    for traj in expert_data:
        for e in traj:
            flatten_expert_data.append(e)

    expert_data = flatten_expert_data

    return expert_data


def main():

    torch.manual_seed(1234)
    np.random.seed(1234)

    os.makedirs("tmp", exist_ok=True)

    expert_data = sample_expert()

    env = make_env()
    ts = env.bundle.task.state
    total_n_iter = int(ts["n_iter_per_ss"] * ts["n_session"])

    n_env = 4
    envs = [make_env for _ in range(n_env)]
    vec_env = SubprocVecEnv(envs)
    vec_env = VecMonitor(vec_env, filename="tmp/log")

    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="./tb/",
                n_steps=total_n_iter,  # This is important to set for the learning to be effective!!
                batch_size=100,
                )

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


if __name__ == "__main__":
    main()

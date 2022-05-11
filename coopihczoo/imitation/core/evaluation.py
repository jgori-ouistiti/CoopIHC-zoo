import numpy as np
import torch

import gym

# from stable_baselines3.common.monitor import Monitor

import coopihc


def evaluate_policy(policy, env, n_eval_episodes=50, deterministic=True):

    if isinstance(env.observation_space, gym.spaces.Dict):
        raise ValueError("Gym observation space should NOT be a dictionary "
                         "(use the filter 'FlattenObservation' from Gym)")

    obs = env.reset()

    n_steps = 0
    ep = 0

    rewards = []

    reward_ep = 0

    with torch.no_grad():
        while True:

            if isinstance(policy, coopihc.BaseAgent):

                _action, _reward = env.unwrapped.bundle.assistant.take_action(increment_turn=False)
                print(_action)
                action = _action
            else:
                action, _states = policy.predict(obs, deterministic=deterministic)

            new_obs, reward, done, info = env.step(action)

            n_steps += 1

            reward_ep += reward

            if done:
                ep += 1
                rewards.append(reward_ep)
                reward_ep = 0

                new_obs = env.reset()

            obs = new_obs

            if ep < n_eval_episodes:
                continue

            break

    return np.mean(rewards), np.std(rewards)

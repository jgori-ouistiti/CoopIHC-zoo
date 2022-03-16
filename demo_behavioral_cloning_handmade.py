import torch
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from coopihczoo.teaching.rl.behavioral_cloning import \
    BC, flatten_trajectories, make_sample_until, RolloutInfoWrapper, rollout

#
# def train_expert(env):
#
#     print("Training a expert.")
#     expert = PPO(
#         policy=MlpPolicy,
#         env=env,
#         seed=0,
#         batch_size=64,
#         ent_coef=0.0,
#         learning_rate=0.0003,
#         n_epochs=10,
#         n_steps=64,
#     )
#     expert.learn(1000)  # Note: change this to 100000 to train a decent expert.
#     return expert


# def sample_expert(env, expert, n_episode=50):
#
#     env = VecMonitor(DummyVecEnv([lambda: env]))
#
#     obs = env.reset()
#
#     n_steps = 0
#     ep = 0
#
#     expert_data = [[], ]
#
#     with torch.no_grad():
#         while ep < n_episode:
#
#             action, _states = expert.predict(obs)
#
#             new_obs, rewards, dones, info = env.step(action)
#
#             n_steps += 1
#
#             expert_data[-1].append({"acts": action.squeeze(), "obs": obs.squeeze()})
#
#             # Handle timeout by bootstraping with value function
#             # see GitHub issue #633
#             for idx, done in enumerate(dones):
#                 if done:
#                     ep += 1
#                     expert_data.append([])
#
#             obs = new_obs
#
#     expert_data = expert_data[:-1]
#
#     return expert_data


def main():

    torch.manual_seed(1234)
    np.random.seed(1234)

    env = gym.make("CartPole-v1")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1000)  # Note: set to 100000 to train a proficient expert

    # expert_data = sample_expert(env=env, expert=expert)
    #
    # np.random.shuffle(expert_data)
    #
    # # Flatten expert data
    # flatten_expert_data = []
    # for traj in expert_data:
    #     for e in traj:
    #         flatten_expert_data.append(e)
    #
    # expert_data = flatten_expert_data

    rollouts = rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        make_sample_until(min_timesteps=None, min_episodes=50),
    )
    transitions = flatten_trajectories(rollouts)

    novice = PPO(
        policy=MlpPolicy,
        env=env,
        # seed=0,
        # batch_size=64,
        # ent_coef=0.0,
        # learning_rate=0.0003,
        # n_epochs=10,
        # n_steps=64,
    )
    policy = novice.policy

    # policy = FeedForward32Policy(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     lr_schedule=ConstantLRSchedule())

    reward, _ = evaluate_policy(policy, Monitor(env), n_eval_episodes=10)
    print(f"Reward before training: {reward}")

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy)

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=1)

    reward, _ = evaluate_policy(bc_trainer.policy, Monitor(env), n_eval_episodes=10)
    print(f"Reward after training: {reward}")


if __name__ == "__main__":
    main()

import torch
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor

from coopihczoo.imitation.core.behavioral_cloning import \
    BC, FeedForward32Policy, ConstantLRSchedule, sample_expert


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

    expert_data = sample_expert(env=env, expert=expert)

    # novice = PPO(
    #     policy=MlpPolicy,
    #     env=env,
    #     # seed=0,
    #     # batch_size=64,
    #     # ent_coef=0.0,
    #     # learning_rate=0.0003,
    #     # n_epochs=10,
    #     # n_steps=64,
    # )
    # policy = novice.policy

    policy = FeedForward32Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=ConstantLRSchedule())

    reward, _ = evaluate_policy(policy, Monitor(env), n_eval_episodes=10)
    print(f"Reward before training: {reward}")

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_data,
        policy=policy)

    print("Training a policy using Behavior Cloning")
    bc_trainer.train()

    reward, _ = evaluate_policy(policy, Monitor(env), n_eval_episodes=10)
    print(f"Reward after training: {reward}")


if __name__ == "__main__":
    main()

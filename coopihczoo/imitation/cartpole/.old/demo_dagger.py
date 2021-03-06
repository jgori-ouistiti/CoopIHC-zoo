import tempfile

import numpy as np
import torch
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer


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

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space)

    reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10, render=False)
    print(f"Reward before training: {reward}")

    # rng = np.random.RandomState(seed=1234)
    torch.manual_seed(1234)
    np.random.seed(1234)

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:

        venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])

        dagger_trainer = SimpleDAggerTrainer(
            venv=venv, scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer)

        dagger_trainer.train(2000)

    torch.manual_seed(1234)
    np.random.seed(1234)
    env = gym.make("CartPole-v1")
    reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10, render=False)
    print(f"Reward after training: {reward}")


if __name__ == "__main__":
    main()

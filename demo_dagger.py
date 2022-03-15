from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import gym

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

import tempfile
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])


bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
)

reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=3, render=True)
print(f"Reward before training: {reward}")

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:

    dagger_trainer = SimpleDAggerTrainer(
        venv=venv, scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer
    )

    dagger_trainer.train(2000)
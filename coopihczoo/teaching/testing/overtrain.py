import os

import torch
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from coopihczoo.teaching.imitation.teaching_bc import make_env

def main():

    os.makedirs("tmp", exist_ok=True)

    evaluate = True

    seed = 1345

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_env = 4

    envs = [make_env for _ in range(n_env)]

    env = SubprocVecEnv(envs)
    env = VecMonitor(env, filename="tmp/log")

    dummy_env = make_env()
    total_n_iter = \
        int(dummy_env.bundle.task.state["n_iter_per_ss"] * dummy_env.bundle.task.state["n_session"])

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./tb/",
                batch_size=total_n_iter*n_env,
                n_steps=total_n_iter)  # This is important to set for the learning to be effective!!

    model.set_parameters('../imitation/tmp/model_1000ep')

    if evaluate:
        env = make_env(seed)
        print("Evaluating...")
        reward, _ = evaluate_policy(model, Monitor(env), n_eval_episodes=50)
        print(f"Reward: {reward}")

    model.learn(total_timesteps=int(1e6))

    env = make_env(seed)
    print("Evaluating...")
    reward, _ = evaluate_policy(model, Monitor(env), n_eval_episodes=50)
    print(f"Reward: {reward}")




if __name__ == "__main__":
    main()
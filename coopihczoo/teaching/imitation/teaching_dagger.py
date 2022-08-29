import numpy as np

from stable_baselines3.common.monitor import Monitor

import torch

from stable_baselines3.common.evaluation import evaluate_policy
from coopihczoo.utils.imitation.run import train_novice_dagger_ppo

from coopihczoo.teaching.imitation.teaching_bc import make_env


def main():

    evaluate_expert = False
    total_timesteps = 5000

    torch.manual_seed(1234)
    np.random.seed(1234)

    env = make_env()

    expert = env.unwrapped.bundle.assistant

    if evaluate_expert:
        print("Evaluating expert...")
        reward, _ = evaluate_policy(expert, Monitor(env), n_eval_episodes=50)
        print(f"Reward expert: {reward}")

    total_n_iter = int(
        env.bundle.task.state["n_iter_per_ss"]
        * env.bundle.task.state["n_session"]
    )

    novice_kwargs = dict(
        policy="MlpPolicy",
        batch_size=total_n_iter,
        n_steps=total_n_iter,
    )

    train_novice_dagger_ppo(
        novice_kwargs=novice_kwargs,
        expert=expert,
        make_env=make_env,
        total_timesteps=total_timesteps
    )


if __name__ == "__main__":
    main()


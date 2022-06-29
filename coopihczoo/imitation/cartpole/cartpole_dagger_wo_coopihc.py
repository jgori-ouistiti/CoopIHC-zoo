import os

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor

from coopihczoo.imitation.core.dagger import DAgger
from cartpole_bc_wo_coopihc import make_env, get_expert_config


def main():

    seed = 123
    expert_total_timesteps = 1e6
    dagger_training_n_episode = 5000

    env_name = "LunarLander-v2"

    expert_kwargs = get_expert_config(seed=seed, env_name=env_name)

    env = make_env(seed=seed, env_name=env_name)
    expert = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert before training: {reward}")

    print("Training the expert...")
    expert.learn(total_timesteps=expert_total_timesteps)  # Note: set to 100000 to train a proficient expert

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    print("Saving the expert...")
    os.makedirs("tmp", exist_ok=True)
    expert.save(f"tmp/lunar_lander_expert_{int(expert_total_timesteps)}")
    # expert.load(f"tmp/lunar_lander_expert_{int(expert_total_timesteps)}")

    env = make_env(seed=seed, env_name=env_name)
    novice = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    dagger_trainer = DAgger(
        env=env,
        expert=expert,
        policy=novice.policy)

    print("Training the novice's policy using dagger...")
    dagger_trainer.train(total_timesteps=dagger_training_n_episode)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")


if __name__ == "__main__":
    main()

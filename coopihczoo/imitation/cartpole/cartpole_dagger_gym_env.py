import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor

from coopihczoo.imitation.core.dagger import DAgger


def make_env(seed):
    env = gym.make("CartPole-v1")
    env.seed(seed=seed)
    return env


def main():

    seed = 123
    expert_total_timesteps = 10000
    dagger_training_n_episode = 5000

    expert_kwargs = dict(
            seed=seed,
            policy='MlpPolicy',
            n_steps=32,
            batch_size=32,
            gae_lambda=0.8,
            gamma=0.98,
            n_epochs=20,
            ent_coef=0.0,
            learning_rate=0.001,
            clip_range=0.2
    )

    env = make_env(seed=seed)
    expert = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert before training: {reward}")

    print("Training the expert...")
    expert.learn(total_timesteps=expert_total_timesteps)  # Note: set to 100000 to train a proficient expert

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    env = make_env(seed=seed)
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

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor

from coopihczoo.imitation.core.behavioral_cloning import BC, sample_expert


from noisy_cartpole import CartPoleModEnv


def make_env(seed):
    # env = gym.make("CartPole-v1")

    # 1 : # v-0 / no noise,
    # 2 : #  5% actuator noise
    # 3 : # 10% actuator noise
    # 4 : #  5% sensor noise
    # 5 : # 10% sensor noise
    # 6 : # 0.1 var sensor noise
    # 7 : # 0.2 var sensor noise
    env = CartPoleModEnv(case=3)
    env.seed(seed=seed)
    return env


def main():

    seed = 123
    expert_total_timesteps = 10000
    sample_expert_n_episode = 50
    sample_expert_n_timestep = None

    expert_kwargs = dict(
        seed=seed,
        policy="MlpPolicy",
        n_steps=32,
        batch_size=32,
        gae_lambda=0.8,
        gamma=0.98,
        n_epochs=20,
        ent_coef=0.0,
        learning_rate=0.001,
        clip_range=0.2,
    )

    env = make_env(seed=seed)
    expert = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert before training: {reward}")

    print("Training the expert...")
    expert.learn(
        total_timesteps=expert_total_timesteps
    )  # Note: set to 100000 to train a proficient expert

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    expert_data = sample_expert(
        env=env,
        expert=expert,
        n_episode=sample_expert_n_episode,
        n_timestep=sample_expert_n_timestep,
    )

    env = make_env(seed=seed)
    novice = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_data,
        policy=novice.policy,
    )

    print("Training the novice's policy using behavior cloning...")
    bc_trainer.train()

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")


if __name__ == "__main__":
    main()

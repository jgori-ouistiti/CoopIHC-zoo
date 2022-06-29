import gym

from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy
from coopihczoo.imitation.core.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor

from coopihczoo.imitation.core.behavioral_cloning import \
    BC, sample_expert


def make_env(seed, env_name):
    env = gym.make(env_name)
    env.seed(seed=seed)
    return env


def get_expert_config(seed, env_name):

    if env_name == "CartPole-v1":

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

    elif env_name == "LunarLander-v2":
        expert_kwargs = dict(
            seed=seed,
            policy='MlpPolicy',
            batch_size=32,
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=3e-4,
            clip_range=0.2
        )

    else:
        raise ValueError

    return expert_kwargs

def main():

    seed = 123
    expert_total_timesteps = 1e5
    sample_expert_n_episode = 50
    sample_expert_n_timestep = None

    env_name = "LunarLander-v2"

    expert_kwargs = get_expert_config(seed, env_name)

    env = make_env(seed=seed, env_name=env_name)
    expert = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert before training: {reward}")

    print("Training the expert...")
    expert.learn(total_timesteps=expert_total_timesteps)  # Note: set to 100000 to train a proficient expert

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    expert_data = sample_expert(env=env, expert=expert,
                                n_episode=sample_expert_n_episode,
                                n_timestep=sample_expert_n_timestep)

    env = make_env(seed=seed, env_name=env_name)
    novice = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_data,
        policy=novice.policy)

    print("Training the novice's policy using behavior cloning...")
    bc_trainer.train()

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")


if __name__ == "__main__":
    main()

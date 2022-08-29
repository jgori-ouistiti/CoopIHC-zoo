import gym

from coopihczoo.utils.imitation.run import run_behavioral_cloning_ppo


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
    expert_total_timesteps = 1e4  # 1e5
    sample_expert_n_episode = 50

    env_name = 'CartPole-v1'  # "LunarLander-v2"  # Can replace with cartpole_and_lunar

    expert_kwargs = get_expert_config(seed, env_name)

    saving_path = f"tmp/{env_name}_bc_wo_coopihc"

    def make_env_no_args():
        return make_env(seed=seed, env_name=env_name)

    run_behavioral_cloning_ppo(
        saving_path=saving_path,
        make_env=make_env_no_args,
        expert_total_timesteps=expert_total_timesteps,
        sample_expert_n_episode=sample_expert_n_episode,
        expert_kwargs=expert_kwargs)


if __name__ == "__main__":
    main()

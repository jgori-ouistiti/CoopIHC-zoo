from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor

from coopihczoo.imitation.core.behavioral_cloning import BC, sample_expert
from coopihczoo.imitation.cartpole.environments.noisy_cartpole import NoisyCartPoleEnv

from coopihczoo.imitation.core.run import run_behavioral_cloning_ppo


def main():

    seed = 123
    expert_total_timesteps = 10000
    sample_expert_n_episode = 50

    saving_path = "tmp/noisy_cartpole"

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

    def make_env():
        env = NoisyCartPoleEnv(scale=1.5)
        env.seed(seed=seed)
        return env

    run_behavioral_cloning_ppo(
        saving_path=saving_path,
        make_env=make_env,
        expert_total_timesteps=expert_total_timesteps,
        sample_expert_n_episode=sample_expert_n_episode,
        expert_kwargs=expert_kwargs)


if __name__ == "__main__":
    main()

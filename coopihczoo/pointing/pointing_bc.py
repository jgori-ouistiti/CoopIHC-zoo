from coopihczoo.pointing.make_env.make_env import make_env
from coopihczoo.utils.imitation.run import run_behavioral_cloning_ppo


def main():

    seed = 123
    sample_expert_n_episode = 1e5
    expert_total_timesteps = 1e5

    config_task = dict(gridsize=12, number_of_targets=3, mode="position")
    config_user = dict(error_rate=0.01)

    expert_kwargs = dict(
        seed=seed,
        policy="MlpPolicy",
    )

    saving_path = "tmp/pointing_bc"

    def make_pointing_env():
        return make_env(seed=seed, config_task=config_task, config_user=config_user)

    run_behavioral_cloning_ppo(
        saving_path=saving_path,
        make_env=make_pointing_env,
        expert_total_timesteps=expert_total_timesteps,
        sample_expert_n_episode=sample_expert_n_episode,
        expert_kwargs=expert_kwargs)


if __name__ == "__main__":
    main()

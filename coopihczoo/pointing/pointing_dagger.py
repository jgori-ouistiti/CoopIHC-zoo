from coopihczoo.utils.imitation.run import run_dagger_ppo
from coopihczoo.pointing.make_env.make_env import make_env


def main():

    seed = 123
    dagger_total_timesteps = 1e5
    expert_total_timesteps = 1e5

    config_task = dict(gridsize=4, number_of_targets=1, mode="position")
    config_user = dict(error_rate=0.01)

    expert_kwargs = dict(
            seed=seed,
            policy='MlpPolicy',
    )

    saving_path = f"tmp/pointing_dagger"

    run_dagger_ppo(
        saving_path=saving_path,
        expert_kwargs=expert_kwargs,
        env=make_env(config_task=config_task,
                      config_user=config_user,
                      seed=seed),
        dagger_total_timesteps=dagger_total_timesteps,
        expert_total_timesteps=expert_total_timesteps
    )


if __name__ == "__main__":
    main()

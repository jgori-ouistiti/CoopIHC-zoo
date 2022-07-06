from coopihczoo.imitation.core.run import run_dagger_ppo
from coopihczoo.imitation.pointing.pointing_bc import make_env


def main():

    seed = 123
    dagger_training_n_episode = 1e5
    expert_total_timesteps = 1e5

    config_task = dict(gridsize=4, number_of_targets=1, mode="position")
    config_user = dict(error_rate=0.01)

    expert_kwargs = dict(
            seed=seed,
            policy='MlpPolicy',
    )

    saving_path = f"tmp/pointing_dagger"


if __name__ == "__main__":
    main()

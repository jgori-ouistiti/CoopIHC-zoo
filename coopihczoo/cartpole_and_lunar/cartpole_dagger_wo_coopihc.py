from cartpole_bc_wo_coopihc import make_env, get_expert_config

from coopihczoo.utils.imitation.run import run_dagger_ppo


def main():

    seed = 123
    expert_total_timesteps = 1e6
    dagger_total_timesteps = 1e4

    eval_expert = True

    env_name = "LunarLander-v2"  # Can replace with cartpole_and_lunar

    expert_kwargs = get_expert_config(seed=seed, env_name=env_name)

    saving_path = f"tmp/{env_name}_dagger_wo_coopihc"

    def make_env_no_args():
        return make_env(seed=seed, env_name=env_name)

    run_dagger_ppo(
        saving_path=saving_path,
        make_env=make_env_no_args,
        expert_total_timesteps=expert_total_timesteps,
        dagger_total_timesteps=dagger_total_timesteps,
        expert_kwargs=expert_kwargs,
        eval_expert=eval_expert)


if __name__ == "__main__":
    main()

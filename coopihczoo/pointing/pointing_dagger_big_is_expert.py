from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from coopihczoo.utils.imitation.run import train_novice_dagger_ppo

from coopihczoo.pointing.make_env.make_env import make_env


def main():

    seed = 123
    dagger_total_timesteps = 1e5

    config_task = dict(gridsize=4, number_of_targets=1, mode="position")
    config_user = dict(error_rate=0.01)

    novice_kwargs = dict(
            seed=seed,
            policy='MlpPolicy',
    )

    def make_pointing_env():
        return make_env(
            config_task=config_task,
            config_user=config_user,
            seed=seed)

    env = make_pointing_env()
    expert = env.unwrapped.bundle.assistant  # BigGain Expert

    print("Evaluating expert...")
    reward, _ = evaluate_policy(expert, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    train_novice_dagger_ppo(
        expert=expert,
        make_env=make_pointing_env,
        total_timesteps=dagger_total_timesteps,
        novice_kwargs=novice_kwargs)

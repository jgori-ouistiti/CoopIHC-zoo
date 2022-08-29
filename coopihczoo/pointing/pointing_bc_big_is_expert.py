from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from coopihczoo.utils.imitation.run import train_novice_behavioral_cloning_ppo
from coopihczoo.utils.imitation.behavioral_cloning import sample_expert

from coopihczoo.pointing.make_env.make_env import make_env


def main():

    seed = 123
    sample_expert_n_episode = 1e4

    config_task = dict(gridsize=12, number_of_targets=3, mode="position")
    config_user = dict(error_rate=0.01)

    novice_kwargs = dict(
        seed=seed,
        policy="MlpPolicy",
        n_steps=32,
        batch_size=32,
        gae_lambda=0.8,
        gamma=0.98,
        n_epochs=20,
        ent_coef=0.0,
        learning_rate=0.001,
        clip_range=0.2
    )

    env = make_env(seed=seed,
                   config_task=config_task,
                   config_user=config_user)

    expert = env.unwrapped.bundle.assistant  # BigGain Expert

    print("Sampling expert...")

    expert_data = sample_expert(
        expert=expert,
        n_episode=sample_expert_n_episode,
        env=env,
        deterministic=True)

    print("Evaluating expert...")
    reward, _ = evaluate_policy(expert, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    train_novice_behavioral_cloning_ppo(
        make_env=lambda: make_env(seed=seed,
                                  config_task=config_task,
                                  config_user=config_user),
        expert_data=expert_data,
        novice_kwargs=novice_kwargs)


if __name__ == "__main__":
    main()

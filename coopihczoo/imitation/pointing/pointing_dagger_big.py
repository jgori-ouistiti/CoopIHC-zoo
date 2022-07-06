import os.path
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from coopihc import Bundle, TrainGym

from coopihczoo.imitation.core.dagger import DAgger

from coopihc.examples.simplepointing.users import CarefulPointer
from coopihc.examples.simplepointing.envs import SimplePointingTask
from coopihc.examples.simplepointing.assistants import BIGGain

from pointing_dagger import AssistantActionWrapper


config_task = dict(gridsize=12, number_of_targets=3, mode="position")
config_user = dict(error_rate=0.01)


obs_keys = (
    "assistant_state__beliefs",
    "task_state__position",
    "task_state__targets",
    "user_action__action",
)


def make_env(seed):

    task = SimplePointingTask(**config_task)
    user = CarefulPointer(**config_user)
    assistant = BIGGain()
    bundle = Bundle(
        seed=seed,
        task=task,
        user=user,
        assistant=assistant,
        random_reset=True,
        reset_turn=3,
        reset_skip_user_step=False,
    )

    env = TrainGym(bundle, train_user=False, train_assistant=True)

    env = FlattenObservation(FilterObservation(env, obs_keys))
    env = AssistantActionWrapper(env)

    # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    check_env(env)
    return env


def main():

    seed = 123
    dagger_training_n_episode = 1e5

    expert_total_timesteps = 2 * 1e4

    saving_path = "train_assistant_100000"

    expert_kwargs = dict(
        seed=seed,
        policy="MlpPolicy",
    )

    config_task = dict(gridsize=4, number_of_targets=1, mode="position")
    config_user = dict(error_rate=0.01)

    env = make_env(seed=seed)

    if not os.path.exists(f"{saving_path}.zip"):

        expert = PPO(env=env, **expert_kwargs)


        reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
        print(f"Reward expert before training: {reward}")

        print("Training the expert...")
        expert.learn(expert_total_timesteps)

        expert.save(saving_path)

    else:
        expert = PPO.load(saving_path)

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    env = make_env(seed=seed)
    novice = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    dagger_trainer = DAgger(env=env, expert=expert, policy=novice.policy)

    print("Training the novice's policy using dagger...")
    dagger_trainer.train(total_timesteps=dagger_training_n_episode)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")


if __name__ == "__main__":
    main()

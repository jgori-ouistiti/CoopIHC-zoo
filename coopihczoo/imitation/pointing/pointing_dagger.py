import numpy as np

from gym import ActionWrapper
from gym.spaces import Box
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


config_task = dict(gridsize=4, number_of_targets=1, mode="position")
config_user = dict(error_rate=0.01)

obs_keys = (
    "assistant_state__beliefs",
    "task_state__position",
    "task_state__targets",
    "user_action__action",
)


class AssistantActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        _as = env.action_space["assistant_action__action"]
        self.action_space = Box(
            low=-1, high=1, shape=_as.shape, dtype=np.float32
        )
        self.low, self.high = _as.low, _as.high
        self.half_amp = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2

    def action(self, action):
        return {"assistant_action__action": int(action*self.half_amp + self.mean)}

    def reverse_action(self, action):
        raw = action["assistant_action__action"]
        return (raw - self.mean) / self.half_amp


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
    dagger_training_n_episode = 5000

    expert_kwargs = dict(
            seed=seed,
            policy='MlpPolicy',
    )

    env = make_env(seed=seed)
    expert = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert before training: {reward}")

    print("Training the expert...")
    expert.learn(10000)  # Note: set to 100000 to train a proficient expert

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    env = make_env(seed=seed)
    novice = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    dagger_trainer = DAgger(
        env=env,
        expert=expert,
        policy=novice.policy)

    print("Training the novice's policy using dagger...")
    dagger_trainer.train(total_timesteps=dagger_training_n_episode)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")


if __name__ == "__main__":
    main()

import numpy as np

from gym import ActionWrapper
from gym.spaces import Box
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3.common.env_checker import check_env

from coopihc import Bundle

from coopihc.examples.simplepointing.users import CarefulPointer
from coopihc.examples.simplepointing.envs import SimplePointingTask
from coopihc.examples.simplepointing.assistants import BIGGain

from coopihczoo.imitation.core.run import run_behavioral_cloning_ppo


def make_env(seed,
             config_task,
             config_user):

    class AssistantActionWrapper(ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            _as = env.action_space["assistant_action__action"]
            self.action_space = Box(low=-1, high=1, shape=_as.shape, dtype=np.float32)
            self.low, self.high = _as.low, _as.high
            self.half_amp = (self.high - self.low) / 2
            self.mean = (self.high + self.low) / 2

        def action(self, action):
            return {"assistant_action__action": int(action * self.half_amp + self.mean)}

        def reverse_action(self, action):
            raw = action["assistant_action__action"]
            return (raw - self.mean) / self.half_amp

    obs_keys = (
        "assistant_state__beliefs",
        "task_state__position",
        "task_state__targets",
        "user_action__action",
    )

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

    env = bundle.convert_to_gym_env(train_user=False, train_assistant=True)

    env = FlattenObservation(FilterObservation(env, obs_keys))
    env = AssistantActionWrapper(env)

    # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    check_env(env)
    return env


def main():

    seed = 123
    sample_expert_n_episode = 1e4
    expert_total_timesteps = 1e5

    config_task = dict(gridsize=4, number_of_targets=1, mode="position")
    config_user = dict(error_rate=0.01)

    expert_kwargs = dict(
        seed=seed,
        policy="MlpPolicy",
    )

    run_behavioral_cloning_ppo(
        saving_path="tmp/pointing_bc",
        make_env=lambda: make_env(seed=seed, config_task=config_task, config_user=config_user),
        expert_total_timesteps=expert_total_timesteps,
        sample_expert_n_episode=sample_expert_n_episode,
        expert_kwargs=expert_kwargs)


if __name__ == "__main__":
    main()

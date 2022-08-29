import numpy as np

from gym import ActionWrapper
from gym.spaces import Box
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3.common.env_checker import check_env

from coopihc import Bundle, WrapperReferencer

from coopihc.examples.simplepointing.users import CarefulPointer
from coopihc.examples.simplepointing.envs import SimplePointingTask
from coopihc.examples.simplepointing.assistants import BIGGain


def make_env(seed,
             config_task,
             config_user):

    class AssistantActionWrapper(ActionWrapper, WrapperReferencer):
        def __init__(self, env):
            ActionWrapper.__init__(self, env)
            WrapperReferencer.__init__(self, env)

            _as = env.action_space["assistant_action__action"]
            self.action_space = Box(low=-1, high=1, shape=_as.shape,
                                    dtype=np.float32)
            self.low, self.high = _as.low, _as.high
            self.half_amp = (self.high - self.low) / 2
            self.mean = (self.high + self.low) / 2

        def action(self, action):
            return {"assistant_action__action": int(action * self.half_amp
                                                    + self.mean)}

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
        reset_go_to=3,
    )

    env = bundle.convert_to_gym_env(train_user=False, train_assistant=True)

    env = FlattenObservation(FilterObservation(env, obs_keys))
    env = AssistantActionWrapper(env)

    # Use env_checker from stable_baselines3 to verify that the make_env adheres to the Gym API
    check_env(env)
    return env

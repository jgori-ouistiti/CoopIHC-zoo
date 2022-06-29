from token import AT
import numpy as np

from gym import ActionWrapper
from gym.spaces import Box
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO

# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from coopihc import Bundle, TrainGym
from coopihc.bundle.wrappers.Train import apply_wrappers

from coopihczoo.imitation.core.behavioral_cloning import BC, sample_expert

from coopihczoo.imitation.core.evaluation import evaluate_policy

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


class WrapperReferencer:
    @property
    def upper_env(self):
        return self.upper_env

    def __init__(self, env):
        env.upper_env = self


class AssistantActionWrapper(ActionWrapper, WrapperReferencer):
    def __init__(self, env):
        ActionWrapper.__init__(self, env)
        WrapperReferencer.__init__(self, env)
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
    print(env.outer_env)
    # env = FlattenObservation(FilterObservation(env, obs_keys))
    env = AssistantActionWrapper(env)
    print(env.outer_env)

    # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    check_env(env)
    return env


# task = SimplePointingTask(**config_task)
# user = CarefulPointer(**config_user)
# assistant = BIGGain()
# bundle = Bundle(
#     seed=None,
#     task=task,
#     user=user,
#     assistant=assistant,
#     random_reset=True,
#     reset_turn=3,
#     reset_skip_user_step=False,
# )

# env = TrainGym(bundle, train_user=False, train_assistant=True)
# new_env = AssistantActionWrapper(env)

env = make_env(None)
exit()
obs = env.reset()
_action, reward = env.bundle.assistant.take_action(increment_turn=False)
assistant = env.bundle.assistant

wrapped_action = apply_wrappers(_action, env)
self = assistant
take_action = self.take_action(agent_observation=obs, increment_turn=False)[0]
another_action = apply_wrappers(take_action, self.bundle.trainer.outer_env)
other_action, state = assistant.predict(obs, increment_turn=False)

# env = make_env(None)
# obs = env.reset()
# _action, reward = env.bundle.assistant.take_action(increment_turn=False)
# assistant = env.bundle.assistant

# action = apply_wrappers(_action, env)

# other_action, state = assistant.predict(obs, increment_turn=False)
exit()


def main():

    seed = 123
    env = make_env(seed=seed)

    expert_sampling_n_episode = 50

    expert = env.unwrapped.bundle.assistant

    reward, _ = evaluate_policy(policy=expert, env=Monitor(env), n_eval_episodes=50)
    print(f"Reward expert: {reward}")

    expert_data = sample_expert(
        env=env, expert=expert, n_episode=expert_sampling_n_episode
    )

    env = make_env(seed=seed)
    novice = PPO(env=env, seed=seed, policy="MlpPolicy")

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_data,
        policy=novice.policy,
    )

    print("Training the novice's policy using behavior cloning...")
    bc_trainer.train()

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")


if __name__ == "__main__":
    main()

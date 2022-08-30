import os

from gym.wrappers import FilterObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from gym import ActionWrapper
from gym.spaces import Discrete

from coopihc import Bundle, TrainGym

from coopihczoo.teaching.users.users_naive_implementation import User
from coopihczoo.teaching.envs.envs_naive_implementation import Task
from coopihczoo.teaching.assistants.assistants_naive_implementation.rl import Teacher
from coopihczoo.teaching.config import config_example


class AssistantActionWrapper(ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(env.action_space["assistant_action__action"].n)

    def action(self, action):
        return {"assistant_action__action": int(action)}

    def reverse_action(self, action):
        return action["assistant_action__action"]


def make_env():

    task = Task(**config_example.task_kwargs)
    user = User(**config_example.user_kwargs)
    assistant = Teacher()
    bundle = Bundle(task=task, user=user, assistant=assistant,
                    random_reset=False,
                    reset_turn=3,
                    reset_skip_user_step=True)  # Begin by assistant

    env = TrainGym(
        bundle,
        train_user=False,
        train_assistant=True)

    # # Use env_checker from stable_baselines3 to verify that the make_env adheres to the Gym API
    # check_env(make_env, warn=False)

    env = FilterObservation(
        env,
        ("assistant_state__memory", "assistant_state__progress"))

    env = AssistantActionWrapper(env)
    return env


def run_rl():

    os.makedirs("tmp", exist_ok=True)

    # make_env = Monitor(make_env, filename="tmp/log")
    # make_env = make_env()

    envs = [make_env for _ in range(4)]

    env = SubprocVecEnv(envs)
    env = VecMonitor(env, filename="tmp/log")

    dummy_env = make_env()
    total_n_iter = \
        int(dummy_env.bundle.task.state["n_iter_per_ss"] * dummy_env.bundle.task.state["n_session"])

    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tb/",
                n_steps=total_n_iter)  # This is important to set for the learning to be effective!!

    model.learn(total_timesteps=int(1e7))
    model.save("saved_model")


if __name__ == "__main__":
    run_rl()

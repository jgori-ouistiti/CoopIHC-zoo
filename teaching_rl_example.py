import os

from gym.wrappers import FilterObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from coopihc import Bundle, TrainGym

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.rl import Teacher
from coopihczoo.teaching.config import config_example
from coopihczoo.teaching.action_wrapper.action_wrapper import AssistantActionWrapper


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

    # # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    # check_env(env, warn=False)

    env = FilterObservation(
        env,
        ("memory", "progress"))

    env = AssistantActionWrapper(env)
    return env


def run_rl():

    os.makedirs("tmp", exist_ok=True)

    # env = Monitor(env, filename="tmp/log")
    # env = make_env()

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

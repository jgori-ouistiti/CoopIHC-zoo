import os

# from gym.wrappers import FilterObservation

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # , DummyVecEnv
# from stable_baselines3.common.env_checker import check_env

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
                    reset_start_after=2,
                    reset_go_to=3)  # Begin by assistant

    filterdict = dict(
        {
            "assistant_state": dict({"progress": ..., "memory": ...}),
        }
    )

    env = TrainGym(
        bundle,
        filter_observation=filterdict,
        train_user=False,
        train_assistant=True)

    env = AssistantActionWrapper(env)

    # env = FilterObservation(
    #     env,
    #     ("memory", "progress", ))

    # # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    # check_env(env, warn=False)

    return env


def run_rl():

    os.makedirs("tmp", exist_ok=True)

    # env = Monitor(env, filename="tmp/log")
    env = make_env()
    ts = env.bundle.task.state
    total_n_iter = int(ts.n_iter_per_ss * ts.n_session)

    n_env = 4
    envs = [make_env for _ in range(n_env)]
    env = SubprocVecEnv(envs)

    # env = DummyVecEnv([make_env])

    env = VecMonitor(env, filename="tmp/log")

    model = A2C("MultiInputPolicy", env, verbose=True, tensorboard_log="./tb/",
                n_steps=total_n_iter)  # ,  # This is important to set for the learning to be effective!!
                # batch_size=100)  #*n_env)

    model.learn(total_timesteps=int(1e7))
    model.save("saved_model")


if __name__ == "__main__":
    run_rl()

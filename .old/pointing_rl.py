import os

from gym.wrappers import FilterObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from coopihczoo.pointing.envs.envs import SimplePointingTask
from coopihczoo.pointing.users.users import CarefulPointer
from coopihczoo.pointing.assistants.assistants import BIGGain

from coopihc import Bundle, TrainGym




#from coopihczoo.teaching.users import User
#from coopihczoo.teaching.envs import Task
#from coopihczoo.teaching.assistants.rl import Teacher
# from coopihczoo.teaching.config import config_example
# from coopihczoo.teaching.action_wrapper.action_wrapper import AssistantActionWrapper



from gym import ActionWrapper
from gym.spaces import Box


class AssistantActionWrapper(ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(

            env.action_space["assistant_action"].n)

    def action(self, action):
        return {"assistant_action": int(action)}

    def reverse_action(self, action):
        return action["assistant_action"]


def make_env():

    task = SimplePointingTask(gridsize=31, number_of_targets=8, mode="position")
    user = CarefulPointer(error_rate=0.05)
    assistant = BIGGain()
    bundle = Bundle(task=task, user=user, assistant=assistant,
                    random_reset=True,
                    reset_turn=3,
                    reset_skip_user_step=False)

    env = TrainGym(
        bundle,
        train_user=False,
        train_assistant=True)

    # # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    # check_env(env, warn=False)

    env = FilterObservation(
        env,
        ("beliefs", "position"))

    env = AssistantActionWrapper(env)
    return env


def run_rl():

    os.makedirs("tmp", exist_ok=True)

    # env = Monitor(env, filename="tmp/log")
    env = make_env()
    print(env.action_space)
    print(env.observation_space)

    # envs = [make_env for _ in range(4)]
    #
    # env = SubprocVecEnv(envs)
    # env = VecMonitor(env, filename="tmp/log")
    #
    # dummy_env = make_env()
    # total_n_iter = \
    #     int(dummy_env.bundle.task.state["n_iter_per_ss"] * dummy_env.bundle.task.state["n_session"])
    #
    # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tb/",
    #             n_steps=total_n_iter)  # This is important to set for the learning to be effective!!
    #
    # model.learn(total_timesteps=int(1e7))
    # model.save("saved_model")


if __name__ == "__main__":
    run_rl()

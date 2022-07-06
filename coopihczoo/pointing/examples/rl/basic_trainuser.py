from stable_baselines3.common.env_checker import check_env

from coopihczoo.pointing.envs import SimplePointingTask
from coopihczoo.pointing.assistants import ConstantCDGain
from coopihczoo.pointing.users import CarefulPointer

from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space

from coopihc.policy.BasePolicy import BasePolicy
from coopihc.bundle.Bundle import Bundle
from coopihc.bundle.wrappers.Train import Train

import numpy

# from stable_baselines import PPO2 as PPO
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines.common import make_vec_env


task = SimplePointingTask(gridsize=31, number_of_targets=8)
unitcdgain = ConstantCDGain(1)
action_state = State(
    **{
        "action": StateElement(
            values=0,
            spaces=Space([numpy.array([-5 + i for i in range(11)], dtype=numpy.int16)]),
        )
    }
)
policy = BasePolicy(action_state)
user = CarefulPointer(override_agent_policy=policy)
bundle = Bundle(task=task, user=user, assistant=unitcdgain)
env = Train(bundle, train_user=True, train_assistant=False, force=True)
# Check make_env is compatible
check_env(env, warn=True, skip_render_check=True)


# With Wrapper provided from
wrapper = env.action_wrappers(env)


# observation_dict = OrderedDict(
# {
#     "task_state": OrderedDict({"position": 0}),
#     "user_state": OrderedDict({"goal": 0}),
# }
# )

# class ThisActionWrapper(gym.ActionWrapper):
# def __init__(self, make_env):
#     super().__init__(make_env)
#     self.N = make_env.action_space[0].n
#     self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

# def action(self, action):
#     return int(
#         numpy.round((action * (self.N - 1e-3) / 2) + (self.N - 1) / 2)[0]
#     )

# md_env = ThisActionWrapper(
# Train(
#     bundle, observation_mode="multidiscrete", observation_dict=observation_dict
# )
# )

# from stable_baselines3.common.env_checker import check_env

# check_env(md_env)

# obs = md_env.reset()
# print(obs)
# print(md_env.bundle.game_state)
# # =============
# def make_env(rank, seed=0):
# def _init():
#     task = SimplePointingTask(gridsize=31, number_of_targets=8)
#     unitcdgain = ConstantCDGain(1)

#     policy = BasePolicy(
#         action_space=[coopihc.space.Discrete(10)],
#         action_set=[[-5 + i for i in range(5)] + [i + 1 for i in range(5)]],
#         action_values=None,
#     )

#     user = CarefulPointer(agent_policy=policy)
#     bundle = PlayUser(task, user, unitcdgain)

#     observation_dict = OrderedDict(
#         {
#             "task_state": OrderedDict({"position": 0}),
#             "user_state": OrderedDict({"goal": 0}),
#         }
#     )
#     make_env = ThisActionWrapper(
#         Train(
#             bundle,
#             observation_mode="multidiscrete",
#             observation_dict=observation_dict,
#         )
#     )

#     make_env.seed(seed + rank)
#     return make_env

# # set_random_seed(seed)
# return _init

# # =============

# if __name__ == "__main__":
# make_env = SubprocVecEnv([make_env(i) for i in range(4)])

# model = PPO("MlpPolicy", make_env, verbose=1, tensorboard_log="./tb/")
# print("start training")
# model.learn(total_timesteps=600000)
# model.save("saved_model")

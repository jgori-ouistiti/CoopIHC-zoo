import gym
from gym.envs.classic_control.cartpole import CartPoleEnv

from gym.wrappers import FlattenObservation, FilterObservation
import numpy as np
import warnings
import copy

from stable_baselines3 import PPO, A2C
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from coopihc import InteractionTask, array_element, Bundle, TrainGym, BaseAgent
from coopihczoo.teaching.rl.behavioral_cloning import \
    BC, FeedForward32Policy, ConstantLRSchedule, sample_expert


class MyCartpole:
    def __init__(self, seed=None):

        super().__init__()
        self.env = gym.make("CartPole-v1")
        self.env.seed(seed)
        self.i = 0

    # def reset(self):
    #     obs = self.env.reset()
    #     return obs

    def step(self, action):
        # print(self.i, action)
        self.i += 1
        return self.env.step(action)

    def __getattr__(self, item):
        if item.startswith('_'):
            return AttributeError
        return getattr(self.env, item)


#
# mcp = MyCartpole()

class CoopIHC_CartPole(InteractionTask):
    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.env = env

        self.state["x"] = array_element(
            init=0,
            low=np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38]),
            high=np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38]),
            shape=(4,),
            out_of_bounds_mode="warning",
            dtype=np.float32)

        self.i = 0
        # self.state['x'] = self.env.state

    def reset(self, dic=None):
        if dic is not None:
            warnings.warn(f"trying to reset task {self.env.__name__} with dictionnary {dic}, "
                          f"but Gym does not support that")
        obs = self.env.reset()
        self.state['x'] = obs
        return obs

    def on_user_action(self, *args, **kwargs):
        pass

    def on_assistant_action(self, *args, assistant_action = None, **kwargs):
        aa = int(assistant_action)
        # print(self.i, aa)
        self.i += 1
        task_state, _reward, done, _ = self.env.step(aa)
        # print(f"reward from cartpole {_reward}")
        # print(f"task_state after env.step: {task_state}")
        self.state['x'] = task_state
        return self.state['x'], _reward, done


class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.bundle.task.env.action_space

    def action(self, action):
        return {"assistant_action__action": action}

    def reverse_action(self, action):
        return action["assistant_action__action"]


def make_env(seed):
    env = gym.make("CartPole-v1")
    coopihc_cartpole = CoopIHC_CartPole(env)
    bundle = Bundle(task=coopihc_cartpole, assistant=BaseAgent('assistant'))

    env = TrainGym(
        bundle,
        # filter_observation=filterdict,
        train_user=False,
        train_assistant=True,
        reset_turn=3,
        # reset_dic= {'task_state':{'x': np.array([ 0.04337885, -0.01720043, -0.03162273,  0.03466498])}},
        filter_observation={'task_state': {'x': slice(0, 4, 1)}})

    env = FlattenObservation(CustomActionWrapper(env))

    env.bundle.task.env = gym.make("CartPole-v1")
    env.bundle.task.env.seed(seed)
    return env


seed = 123
expert_kwargs = dict(
        seed=seed,
        policy='MlpPolicy',
        n_steps=32,
        batch_size=32,
        gae_lambda=0.8,
        gamma=0.98,
        n_epochs=20,
        ent_coef=0.0,
        learning_rate=0.001,
        clip_range=0.2
)

# env = MyCartpole(seed=seed)# gym.make("CartPole-v1")
# # env.seed(seed)
#
# expert = PPO(env=env, **expert_kwargs)
#
# reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=10)
# print(f"Reward expert before training: {reward}")
#
# expert.learn(10000)  # Note: set to 100000 to train a proficient expert
#
# reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
# print(f"Reward expert after training: {reward}")

# Reward expert before training: 9.74
# Reward expert after training: 500.0


env = make_env(seed=seed)
expert = PPO(env=env, **expert_kwargs)

reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
print(f"Reward expert before training: {reward}")

expert.learn(10000)  # Note: set to 100000 to train a proficient expert

reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
print(f"Reward expert after training: {reward}")

expert_data = sample_expert(env=env, expert=expert)

    # novice = PPO(
    #     policy=MlpPolicy,
    #     env=env,
    #     # seed=0,
    #     # batch_size=64,
    #     # ent_coef=0.0,
    #     # learning_rate=0.0003,
    #     # n_epochs=10,
    #     # n_steps=64,
    # )
    # policy = novice.policy

env = make_env(seed=seed)
novice = PPO(env=env, **expert_kwargs)

# policy = FeedForward32Policy(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     lr_schedule=ConstantLRSchedule())

reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
print(f"Reward before training: {reward}")

bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=expert_data,
    policy=novice.policy)

print("Training a policy using Behavior Cloning")
bc_trainer.train()

reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
print(f"Reward after training: {reward}")
#
# # env = AssistantActionWrapper(env)

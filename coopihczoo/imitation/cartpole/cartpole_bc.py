import numpy as np
import warnings

import gym
from gym.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from coopihc import \
    InteractionTask, array_element, Bundle, TrainGym, BaseAgent
from coopihczoo.imitation.core.behavioral_cloning import \
    BC, sample_expert


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
        task_state, _reward, done, _ = self.env.step(aa)
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
        train_user=False,
        train_assistant=True,
        reset_turn=3,
        filter_observation={'task_state': {'x': ...}})

    env = FlattenObservation(CustomActionWrapper(env))

    env.bundle.task.env = gym.make("CartPole-v1")
    env.bundle.task.env.seed(seed)
    return env


def main():

    seed = 123
    expert_total_timesteps = 10000

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

    env = make_env(seed=seed)
    expert = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert before training: {reward}")

    print("Training the expert...")
    expert.learn(expert_total_timesteps)  # Note: set to 100000 to train a proficient expert

    reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward expert after training: {reward}")

    expert_data = sample_expert(env=env, expert=expert)

    env = make_env(seed=seed)
    novice = PPO(env=env, **expert_kwargs)

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_data,
        policy=novice.policy)

    print("Training the novice's policy using behavior cloning...")
    bc_trainer.train()

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")


if __name__ == "__main__":
    main()

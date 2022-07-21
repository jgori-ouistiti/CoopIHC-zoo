import numpy as np
import warnings

import gym
from gym.wrappers import FlattenObservation

from coopihc import \
    InteractionTask, array_element, Bundle, TrainGym, BaseAgent

from coopihczoo.utils.imitation.run import run_behavioral_cloning_ppo


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

    # Not necessary in the general case but here, we want the same seed
    env.bundle.task.env = gym.make("CartPole-v1")
    env.bundle.task.env.seed(seed)
    return env


def main():

    seed = 123
    expert_total_timesteps = 10000

    sample_expert_n_episode = 5000

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

    saving_path = "tmp/cartpole_bc"

    run_behavioral_cloning_ppo(
        saving_path=saving_path,
        make_env=lambda: make_env(seed=seed),
        expert_total_timesteps=expert_total_timesteps,
        sample_expert_n_episode=sample_expert_n_episode,
        expert_kwargs=expert_kwargs)


if __name__ == "__main__":
    main()

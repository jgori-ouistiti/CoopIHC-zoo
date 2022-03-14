import torch
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor


import os

from gym.wrappers import FilterObservation
# import gym

from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from coopihc import Bundle, TrainGym

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.rl import Teacher

from coopihczoo.teaching.config import config_example
from coopihczoo.teaching.assistants.conservative_sampling_expert import ConservativeSamplingExpert

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
        train_assistant=True,
    )

    env = FilterObservation(
        env,
        ("memory", "progress"))

    env = AssistantActionWrapper(env)
    return env


class BC:

    def __init__(
        self,
        policy,
        observation_space: gym.Space,
        action_space: gym.Space,
        expert_data,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        shuffle_data: bool = False,
        batch_size: int = 32
    ):
        """Behavioral cloning (BC).

        Recovers a policy via supervised learning on observation-action Tensor
        pairs, sampled from a Torch DataLoader or any Iterator that ducktypes
        `torch.utils.data.DataLoader`.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            expert_data: If not None, then immediately call
                  `self.set_expert_data_loader(expert_data)` during initialization.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                  weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
        """
        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")

        self.action_space = action_space
        self.observation_space = observation_space

        self.policy = policy

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(self.policy.parameters(), **optimizer_kwargs)

        self.expert_data_loader = None
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

        self.expert_data_loader = self._get_expert_data_loader(expert_data,
                                                               shuffle=shuffle_data,
                                                               batch_size=batch_size)

    @staticmethod
    def _get_expert_data_loader(expert_data, shuffle, batch_size):

        return torch.utils.data.DataLoader(
            expert_data,
            shuffle=shuffle,
            batch_size=batch_size)

    def _calculate_loss(
        self,
        obs,   #: Union[th.Tensor, np.ndarray],
        acts,  #: Union[th.Tensor, np.ndarray],
    ): #-> Tuple[th.Tensor, Dict[str, float]]:
        """
        Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            obs: The observations seen by the expert. If this is a Tensor, then
                gradients are detached first before loss is calculated.
            acts: The actions taken by the expert. If this is a Tensor, then its
                gradients are detached first before loss is calculated.

        Returns:
            loss: The supervised learning loss for the behavioral clone to optimize.
            stats_dict: Statistics about the learning process to be logged.

        """
        # obs = obs.detach()
        # acts = acts.detach()

        _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        l2_norms = [torch.sum(torch.square(w)) for w in self.policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        stats_dict = dict(
            neglogp=neglogp.item(),
            loss=loss.item(),
            entropy=entropy.item(),
            ent_loss=ent_loss.item(),
            prob_true_act=prob_true_act.item(),
            l2_norm=l2_norm.item(),
            l2_loss=l2_loss.item(),
        )

        return loss, stats_dict

    def train(self, n_epochs: int = 100):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`.

        Args:
            n_epochs: Number of complete passes made through dataset.
        """
        samples_so_far = 0
        batch_num = 0
        for epoch_num in range(n_epochs):
            for batch in self.expert_data_loader:

                batch_num += 1
                batch_size = len(batch["obs"])
                assert batch_size > 0
                samples_so_far += batch_size

                loss, stats_dict = self._calculate_loss(batch["obs"], batch["acts"])

                self.optimizer.zero_grad()
                loss.backward()

                for p in self.policy.parameters():
                    print(p.grad)

                self.optimizer.step()
                stats_dict["epoch_num"] = epoch_num
                stats_dict["n_updates"] = batch_num
                stats_dict["batch_size"] = batch_size

    def save_policy(self, policy_path: str) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        torch.save(self.policy, policy_path)


def sample_expert():

    task = Task(**config_example.task_kwargs)
    user = User(**config_example.user_kwargs)
    assistant = ConservativeSamplingExpert()
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False)
    bundle.reset(
        turn=3, skip_user_step=True
    )  # Reset in a state where the user has already produced an observation and made an inference.

    obs = assistant.state

    expert_data = [[], ]

    while True:

        state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
        new_obs = state["assistant_state"]
        action = int(state['assistant_action']["action"])

        obs_dic = {"memory": obs["memory"].view(np.ndarray),
                   "progress": obs["progress"].view(np.ndarray)}

        expert_data[-1].append({"acts": action,            # .squeeze(),
                                "obs": obs_dic})    # .squeeze()})

        obs = new_obs
        if is_done:
            break

    # print("Final reward", rewards['first_task_reward'])

    # env = VecMonitor(DummyVecEnv([lambda: env]))
    #
    # obs = env.reset()
    #
    # n_steps = 0
    # ep = 0
    #
    # expert_data = [[], ]
    #
    # with torch.no_grad():
    #     while ep < n_episode:
    #
    #         action, _states = expert.predict(obs)
    #
    #         new_obs, rewards, dones, info = env.step(action)
    #
    #         n_steps += 1
    #
    #         expert_data[-1].append({"acts": action.squeeze(), "obs": obs.squeeze()})
    #
    #         # Handle timeout by bootstraping with value function
    #         # see GitHub issue #633
    #         for idx, done in enumerate(dones):
    #             if done:
    #                 ep += 1
    #                 expert_data.append([])
    #
    #         obs = new_obs

    # expert_data = expert_data[:-1]

    return expert_data


def main():

    os.makedirs("tmp", exist_ok=True)

    # env = make_env()
    # new_obs, rewards, dones, info = env.step(0)
    # print(new_obs)

    expert_data = sample_expert()

    # env = Monitor(env, filename="tmp/log")
    # env = make_env()

    envs = [make_env for _ in range(4)]

    vec_env = SubprocVecEnv(envs)
    vec_env = VecMonitor(vec_env, filename="tmp/log")

    env = make_env()
    total_n_iter = \
        int(env.bundle.task.state["n_iter_per_ss"] * env.bundle.task.state["n_session"])

    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="./tb/",
                n_steps=total_n_iter)  # This is important to set for the learning to be effective!!

    policy = model.policy

    reward, _ = evaluate_policy(policy, Monitor(env), n_eval_episodes=3, render=False)
    print(f"Reward before training: {reward}")

    np.random.shuffle(expert_data)

    # Flatten expert data
    flatten_expert_data = []
    for traj in expert_data:
        for e in traj:
            flatten_expert_data.append(e)

    expert_data = flatten_expert_data

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        expert_data=expert_data,
        policy=policy)

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=10000)

    reward, _ = evaluate_policy(bc_trainer.policy, Monitor(env), n_eval_episodes=3, render=False)
    print(f"Reward after training: {reward}")


if __name__ == "__main__":
    main()

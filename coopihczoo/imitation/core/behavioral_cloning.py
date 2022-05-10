from typing import Any, Mapping, Optional, Tuple, Type, Union
import numpy as np
import torch
import torch.utils.data as th_data
import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common import policies, utils
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv


def sample_expert(env, expert, n_episode=50, n_timestep=None, deterministic=False):

    if isinstance(env.observation_space, gym.spaces.Dict):
        raise ValueError("Gym observation space should NOT be a dictionary "
                         "(use the filter 'FlattenObservation' from Gym)")

    env = VecMonitor(DummyVecEnv([lambda: env]))

    obs = env.reset()

    n_steps = 0
    ep = 0

    expert_data = [[], ]

    with torch.no_grad():
        while True:

            action, _states = expert.predict(obs, deterministic=deterministic)

            new_obs, rewards, dones, info = env.step(action)

            n_steps += 1

            expert_data[-1].append({"acts": action.squeeze(), "obs": obs.squeeze()})

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if done:
                    ep += 1
                    expert_data.append([])

            obs = new_obs

            if n_episode is not None and ep < n_episode:
                continue

            if n_timestep is not None and n_steps < n_timestep:
                continue

            break

    expert_data = expert_data[:-1]

    np.random.shuffle(expert_data)

    # Flatten expert data
    flatten_expert_data = []
    for traj in expert_data:
        for e in traj:
            flatten_expert_data.append(e)

    expert_data = flatten_expert_data

    return expert_data


class FeedForward32Policy(ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32])


class ConstantLRSchedule:
    """A callable that returns a constant learning rate."""

    def __init__(self, lr: float = 1e-3):
        """
        Args:
            lr: the constant learning rate that calls to this object will return.
        """
        self.lr = lr

    def __call__(self, _):
        """
        Returns the constant learning rate.
        """
        return self.lr


class BC:
    """Behavioral cloning (BC).

    Recovers a policy via supervised learning from observation-action pairs.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Optional[policies.ActorCriticPolicy] = None,
        demonstrations=None,
        batch_size: int = 32,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, torch.device] = "auto",
    ):
        """Builds BC.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            policy: a Stable Baselines3 policy; if unspecified,
                defaults to `FeedForward32Policy`.
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            batch_size: The number of samples in each batch of expert data.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.

        Raises:
            ValueError: If `weight_decay` is specified in `optimizer_kwargs` (use the
                parameter `l2_weight` instead.)
        """
        self._demo_data_loader = None
        self.batch_size = batch_size

        if demonstrations is not None:
            self.set_demonstrations(demonstrations)

        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")

        self.action_space = action_space
        self.observation_space = observation_space
        self.device = utils.get_device(device)

        if policy is None:
            policy = FeedForward32Policy(
                observation_space=observation_space,
                action_space=action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=ConstantLRSchedule(torch.finfo(torch.float32).max),
            )
        self.policy = policy.to(self.device)

        assert self.policy.observation_space == self.observation_space
        assert self.policy.action_space == self.action_space

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )

        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

    def set_demonstrations(self, demonstrations, shuffle=True):  # algo_base.AnyTransitions) -> None:

        self._demo_data_loader = th_data.DataLoader(
            demonstrations,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True)

    def _calculate_loss(
        self,
        obs: Union[torch.Tensor, np.ndarray, dict],
        acts: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, Mapping[str, float]]:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            obs: The observations seen by the expert. If this is a Tensor, then
                gradients are detached first before loss is calculated.
            acts: The actions taken by the expert. If this is a Tensor, then its
                gradients are detached first before loss is calculated.

        Returns:
            loss: The supervised learning loss for the behavioral clone to optimize.
            stats_dict: Statistics about the learning process to be logged.

        """
        # obs = torch.as_tensor(obs, device=self.device).detach()
        # acts = torch.as_tensor(acts, device=self.device).detach()

        _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
        # prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        l2_norms = [torch.sum(torch.square(w)) for w in self.policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        # stats_dict = dict(
        #     neglogp=neglogp.item(),
        #     loss=loss.item(),
        #     entropy=entropy.item(),
        #     ent_loss=ent_loss.item(),
        #     prob_true_act=prob_true_act.item(),
        #     l2_norm=l2_norm.item(),
        #     l2_loss=l2_loss.item(),
        # )

        return loss  # , stats_dict

    def train(self):

        """Train with supervised learning.
        """
        for batch in self._demo_data_loader:
            loss = self._calculate_loss(batch["obs"], batch["acts"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_policy(self, policy_path):  # : types.AnyPath) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        torch.save(self.policy, policy_path)

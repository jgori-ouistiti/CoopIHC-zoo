import torch
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor


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

    DEFAULT_BATCH_SIZE: int = 32
    """Default batch size for DataLoader automatically constructed from Transitions.

    See `set_expert_data_loader()`.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        policy_class,# : Type[policies.BasePolicy] = base.FeedForward32Policy,
        expert_data,#: Union[Iterable[Mapping], types.TransitionsMinimal, None] = None,
        policy_kwargs = None,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: str = "cpu",
    ):
        """Behavioral cloning (BC).

        Recovers a policy via supervised learning on observation-action Tensor
        pairs, sampled from a Torch DataLoader or any Iterator that ducktypes
        `torch.utils.data.DataLoader`.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            policy_class: used to instantiate imitation policy.
            policy_kwargs: keyword arguments passed to policy's constructor.
            expert_data: If not None, then immediately call
                  `self.set_expert_data_loader(expert_data)` during initialization.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                  weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.
        """
        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")

        self.action_space = action_space
        self.observation_space = observation_space
        self.policy_class = policy_class
        self.device = device
        self.policy_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=ConstantLRSchedule(),
            # device=self.device,
        )
        self.policy_kwargs.update(policy_kwargs or {})

        self.policy = self.policy_class(**self.policy_kwargs).to(
            self.device
        )  # pytype: disable=not-instantiable
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(self.policy.parameters(), **optimizer_kwargs)

        self.expert_data_loader = None
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

        self.set_expert_data_loader(expert_data)

    def set_expert_data_loader(self, expert_data) -> None:
        """Set the expert data loader, which yields batches of obs-act pairs.

        Changing the expert data loader on-demand is useful for DAgger and other
        interactive algorithms.

        Args:
             expert_data: Either a Torch `DataLoader`, any other iterator that
                yields dictionaries containing "obs" and "acts" Tensors or Numpy arrays,
                or a `TransitionsMinimal` instance.

                If this is a `TransitionsMinimal` instance, then it is automatically
                converted into a shuffled `DataLoader` with batch size
                `BC.DEFAULT_BATCH_SIZE`.
        """

        self.expert_data_loader = torch.utils.data.DataLoader(
            expert_data,
            shuffle=False,
            batch_size=BC.DEFAULT_BATCH_SIZE)

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
        obs = torch.as_tensor(obs, device=self.device).detach()
        acts = torch.as_tensor(acts, device=self.device).detach()

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


def train_expert(env):
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1000)  # Note: change this to 100000 to trian a decent expert.
    return expert


def sample_expert(env, expert, n_episode=50):
    env = VecMonitor(DummyVecEnv([lambda: env]))

    obs = env.reset()

    n_steps = 0
    ep = 0

    expert_data = [[], ]

    with torch.no_grad():
        while ep < n_episode:

            action, _states = expert.predict(obs)

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

    expert_data = expert_data[:-1]

    return expert_data


def main():
    env = gym.make("CartPole-v1")

    expert = train_expert(env)

    expert_data = sample_expert(env=env, expert=expert)

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
        policy_class=FeedForward32Policy
    )

    reward, _ = evaluate_policy(bc_trainer.policy, Monitor(env), n_eval_episodes=10, render=False)
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=1)

    reward, _ = evaluate_policy(bc_trainer.policy, Monitor(env), n_eval_episodes=10, render=False)
    print(f"Reward after training: {reward}")


if __name__ == "__main__":
    main()

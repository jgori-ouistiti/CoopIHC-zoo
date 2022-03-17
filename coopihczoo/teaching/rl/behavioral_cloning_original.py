import contextlib
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Type, Union, List, Sequence
import collections
import warnings
import dataclasses

import numpy as np
import torch
import torch.utils.data as th_data
import gym
import tqdm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common import vec_env, policies, utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


def unwrap_traj(traj):  #: types.TrajectoryWithRew) -> types.TrajectoryWithRew:
    """Uses `RolloutInfoWrapper`-captured `obs` and `rews` to replace fields.

    This can be useful for bypassing other wrappers to retrieve the original
    `obs` and `rews`.

    Fails if `infos` is None or if the trajectory was generated from an
    environment without imitation.util.rollout.RolloutInfoWrapper

    Args:
        traj: A trajectory generated from `RolloutInfoWrapper`-wrapped Environments.

    Returns:
        A copy of `traj` with replaced `obs` and `rews` fields.
    """
    ep_info = traj.infos[-1]["rollout"]
    res = dataclasses.replace(traj, obs=ep_info["obs"], rews=ep_info["rews"])
    assert len(res.obs) == len(res.acts) + 1
    assert len(res.rews) == len(res.acts)
    return res


def rollout(
    policy, #: AnyPolicy,
    venv, #,: VecEnv,
    sample_until, #: GenTrajTerminationFn,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    # verbose: bool = True,
    **kwargs,
): #-> Sequence[types.TrajectoryWithRew]:
    """Generate policy rollouts.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments.
        sample_until: End condition for rollout sampling.
        unwrap: If True, then save original observations and rewards (instead of
            potentially wrapped observations and rewards) by calling
            `unwrap_traj()`.
        exclude_infos: If True, then exclude `infos` from pickle by setting
            this field to None. Excluding `infos` can save a lot of space during
            pickles.
        verbose: If True, then print out rollout stats before saving.
        **kwargs: Passed through to `generate_trajectories`.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    trajs = generate_trajectories(policy, venv, sample_until, **kwargs)
    if unwrap:
        trajs = [unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    # if verbose:
    #     stats = rollout_stats(trajs)
    #     logging.info(f"Rollout stats: {stats}")
    return trajs


class RolloutInfoWrapper(gym.Wrapper):
    """Add the entire episode's rewards and observations to `info` at episode end.

    Whenever done=True, `info["rollouts"]` is a dict with keys "obs" and "rews", whose
    corresponding values hold the NumPy arrays containing the raw observations and
    rewards seen during this episode.
    """

    def __init__(self, env: gym.Env):
        """Builds RolloutInfoWrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._obs = None
        self._rews = None

    def reset(self, **kwargs):
        new_obs = super().reset()
        self._obs = [new_obs]
        self._rews = []
        return new_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._obs.append(obs)
        self._rews.append(rew)

        if done:
            assert "rollout" not in info
            info["rollout"] = {
                "obs": np.stack(self._obs),
                "rews": np.stack(self._rews),
            }
        return obs, rew, done, info


def make_min_timesteps(n: int):  # -> GenTrajTerminationFn:
    """Terminate at the first episode after collecting n timesteps of data.

    Args:
        n: Minimum number of timesteps of data to collect.
            May overshoot to nearest episode boundary.

    Returns:
        A function implementing this termination condition.
    """
    assert n >= 1

    def f(trajectories):  # : Sequence[types.TrajectoryWithRew]):
        timesteps = sum(len(t.obs) - 1 for t in trajectories)
        return timesteps >= n

    return f


def make_sample_until(
    min_timesteps: Optional[int],
    min_episodes: Optional[int],
):  # -> GenTrajTerminationFn:
    """Returns a termination condition sampling for a number of timesteps and episodes.

    Args:
        min_timesteps: Sampling will not stop until there are at least this many
            timesteps.
        min_episodes: Sampling will not stop until there are at least this many
            episodes.

    Returns:
        A termination condition.

    Raises:
        ValueError: Neither of n_timesteps and n_episodes are set, or either are
            non-positive.
    """
    if min_timesteps is None and min_episodes is None:
        raise ValueError(
            "At least one of min_timesteps and min_episodes needs to be non-None",
        )

    conditions = []
    if min_timesteps is not None:
        if min_timesteps <= 0:
            raise ValueError(
                f"min_timesteps={min_timesteps} if provided must be positive",
            )
        conditions.append(make_min_timesteps(min_timesteps))

    if min_episodes is not None:
        if min_episodes <= 0:
            raise ValueError(
                f"min_episodes={min_episodes} if provided must be positive",
            )
        conditions.append(make_min_episodes(min_episodes))

    def sample_until(trajs):  # : Sequence[types.TrajectoryWithRew]) -> bool:
        for cond in conditions:
            if not cond(trajs):
                return False
        return True

    return sample_until


def dataclass_quick_asdict(obj):  # -> Dict[str, Any]:
    """Extract dataclass to items using `dataclasses.fields` + dict comprehension.

    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.

    Args:
        obj: A dataclass instance.

    Returns:
        A dictionary mapping from `obj` field names to values.
    """
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


class _NoopTqdm:
    """Dummy replacement for tqdm.tqdm() when we don't want a progress bar visible."""

    def close(self):
        pass

    def set_description(self, s):
        pass

    def update(self, n):
        pass


class _WrappedDataLoader:
    """Wraps a data loader (batch iterable) and checks for specified batch size."""

    def __init__(
        self,
        data_loader,  #: Iterable[TransitionMapping],
        expected_batch_size: int,
    ):
        """Builds _WrapedDataLoader.

        Args:
            data_loader: The data loader (batch iterable) to wrap.
            expected_batch_size: The batch size to check for.
        """
        self.data_loader = data_loader
        self.expected_batch_size = expected_batch_size

    def __iter__(self):
        """Iterator yielding data from `self.data_loader`, checking `self.expected_batch_size`.

        Yields:
            Identity -- yields same batches as from `self.data_loader`.

        Raises:
            ValueError: `self.data_loader` returns a batch of size not equal to
                `self.expected_batch_size`.
        """
        for batch in self.data_loader:
            if len(batch["obs"]) != self.expected_batch_size:
                raise ValueError(
                    f"Expected batch size {self.expected_batch_size} "
                    f"!= {len(batch['obs'])} = len(batch['obs'])",
                )
            if len(batch["acts"]) != self.expected_batch_size:
                raise ValueError(
                    f"Expected batch size {self.expected_batch_size} "
                    f"!= {len(batch['acts'])} = len(batch['acts'])",
                )
            yield batch


def check_for_correct_spaces(env,  #: GymEnv,
                             observation_space,  # : gym.spaces.Space,
                             action_space):  # gym.spaces.Space) -> None:
    """
    Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
    spaces match after loading the model with given env.
    Checked parameters:
    - observation_space
    - action_space

    :param env: Environment to check for valid spaces
    :param observation_space: Observation space to check against
    :param action_space: Action space to check against
    """
    if observation_space != env.observation_space:
        raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
    if action_space != env.action_space:
        raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")


def _policy_to_callable(
    policy,  # : AnyPolicy,
    venv,    # : VecEnv,
    deterministic_policy: bool,
): #  -> PolicyCallable:
    """Converts any policy-like object into a function from observations to actions."""
    if policy is None:

        def get_actions(states):
            acts = [venv.action_space.sample() for _ in range(len(states))]
            return np.stack(acts, axis=0)

    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):
        # There's an important subtlety here: BaseAlgorithm and BasePolicy
        # are themselves Callable (which we check next). But in their case,
        # we want to use the .predict() method, rather than __call__()
        # (which would call .forward()). So this elif clause must come first!

        def get_actions(states):
            # pytype doesn't seem to understand that policy is a BaseAlgorithm
            # or BasePolicy here, rather than a Callable
            acts, _ = policy.predict(  # pytype: disable=attribute-error
                states,
                deterministic=deterministic_policy,
            )
            return acts

    elif isinstance(policy, Callable):
        get_actions = policy

    else:
        raise TypeError(
            "Policy must be None, a stable-baselines policy or algorithm, "
            f"or a Callable, got {type(policy)} instead",
        )

    if isinstance(policy, BaseAlgorithm):
        # check that the observation and action spaces of policy and environment match
        check_for_correct_spaces(venv, policy.observation_space, policy.action_space)

    return get_actions


@dataclasses.dataclass(frozen=True)
class TransitionsMinimal(th_data.Dataset):
    """A Torch-compatible `Dataset` of obs-act transitions.

    This class and its subclasses are usually instantiated via
    `imitation.data.rollout.flatten_trajectories`.

    Indexing an instance `trans` of TransitionsMinimal with an integer `i`
    returns the `i`th `Dict[str, np.ndarray]` sample, whose keys are the field
    names of each dataclass field and whose values are the ith elements of each field
    value.

    Slicing returns a possibly empty instance of `TransitionsMinimal` where each
    field has been sliced.
    """

    obs: np.ndarray
    """
    Previous observations. Shape: (batch_size, ) + observation_shape.

    The i'th observation `obs[i]` in this array is the observation seen
    by the agent when choosing action `acts[i]`. `obs[i]` is not required to
    be from the timestep preceding `obs[i+1]`.
    """

    acts: np.ndarray
    """Actions. Shape: (batch_size,) + action_shape."""

    infos: np.ndarray
    """Array of info dicts. Shape: (batch_size,)."""

    def __len__(self):
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring.

        Also make array values read-only.

        Raises:
            ValueError: if batch size (array length) is inconsistent
                between `obs`, `acts` and `infos`.
        """
        for val in vars(self).values():
            if isinstance(val, np.ndarray):
                val.setflags(write=False)

        if len(self.obs) != len(self.acts):
            raise ValueError(
                "obs and acts must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.acts)}",
            )

        if len(self.infos) != len(self.obs):
            raise ValueError(
                "obs and infos must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.infos)}",
            )

    # TODO(adam): uncomment below once pytype bug fixed in
    # issue https://github.com/google/pytype/issues/1108
    # @overload
    # def __getitem__(self: T, key: slice) -> T:
    #     pass  # pragma: no cover
    #
    # @overload
    # def __getitem__(self, key: int) -> Mapping[str, np.ndarray]:
    #     pass  # pragma: no cover

    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics."""
        d = dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items()}

        if isinstance(key, slice):
            # Return type is the same as this dataclass. Replace field value with
            # slices.
            return dataclasses.replace(self, **d_item)
        else:
            assert isinstance(key, int)
            # Return type is a dictionary. Array values have no batch dimension.
            #
            # Dictionary of np.ndarray values is a convenient
            # torch.util.data.Dataset return type, as a torch.util.data.DataLoader
            # taking in this `Dataset` as its first argument knows how to
            # automatically concatenate several dictionaries together to make
            # a single dictionary batch with `torch.Tensor` values.
            return d_item


@dataclasses.dataclass(frozen=True)
class Transitions(TransitionsMinimal):
    """A batch of obs-act-obs-done transitions."""

    next_obs: np.ndarray
    """New observation. Shape: (batch_size, ) + observation_shape.

    The i'th observation `next_obs[i]` in this array is the observation
    after the agent has taken action `acts[i]`.

    Invariants:
        * `next_obs.dtype == obs.dtype`
        * `len(next_obs) == len(obs)`
    """

    dones: np.ndarray
    """
    Boolean array indicating episode termination. Shape: (batch_size, ).

    `done[i]` is true iff `next_obs[i]` the last observation of an episode.
    """

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring."""
        super().__post_init__()
        if self.obs.shape != self.next_obs.shape:
            raise ValueError(
                "obs and next_obs must have same shape: "
                f"{self.obs.shape} != {self.next_obs.shape}",
            )
        if self.obs.dtype != self.next_obs.dtype:
            raise ValueError(
                "obs and next_obs must have the same dtype: "
                f"{self.obs.dtype} != {self.next_obs.dtype}",
            )
        if self.dones.shape != (len(self.acts),):
            raise ValueError(
                "dones must be 1D array, one entry for each timestep: "
                f"{self.dones.shape} != ({len(self.acts)},)",
            )
        if self.dones.dtype != bool:
            raise ValueError(f"dones must be boolean, not {self.dones.dtype}")


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    infos: Optional[np.ndarray]
    """An array of info dicts, length trajectory_len."""

    terminal: bool
    """Does this trajectory (fragment) end in a terminal state?

    Episodes are always terminal. Trajectory fragments are also terminal when they
    contain the final state of an episode (even if missing the start of the episode).
    """

    def __len__(self):
        """Returns number of transitions, equal to the number of actions."""
        return len(self.acts)

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.obs) != len(self.acts) + 1:
            raise ValueError(
                "expected one more observations than actions: "
                f"{len(self.obs)} != {len(self.acts)} + 1",
            )
        if self.infos is not None and len(self.infos) != len(self.acts):
            raise ValueError(
                "infos when present must be present for each action: "
                f"{len(self.infos)} != {len(self.acts)}",
            )
        if len(self.acts) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")

    def __setstate__(self, state):
        if "terminal" not in state:
            warnings.warn(
                "Loading old version of Trajectory."
                "Support for this will be removed in future versions.",
                DeprecationWarning,
            )
            state["terminal"] = True
        self.__dict__.update(state)


@dataclasses.dataclass(frozen=True)
class TrajectoryWithRew(Trajectory):
    """A `Trajectory` that additionally includes reward information."""

    rews: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


def _rews_validation(rews: np.ndarray, acts: np.ndarray):
    if rews.shape != (len(acts),):
        raise ValueError(
            "rewards must be 1D array, one entry for each action: "
            f"{rews.shape} != ({len(acts)},)",
        )
    if not np.issubdtype(rews.dtype, np.floating):
        raise ValueError(f"rewards dtype {rews.dtype} not a float")


class TrajectoryAccumulator:
    """Accumulates trajectories step-by-step.

    Useful for collecting completed trajectories while ignoring partially-completed
    trajectories (e.g. when rolling out a VecEnv to collect a set number of
    transitions). Each in-progress trajectory is identified by a 'key', which enables
    several independent trajectories to be collected at once. They key can also be left
    at its default value of `None` if you only wish to collect one trajectory.
    """

    def __init__(self):
        """Initialise the trajectory accumulator."""
        self.partial_trajectories = collections.defaultdict(list)

    def add_step(
        self,
        step_dict: Mapping[str, np.ndarray],
        key  #: Hashable = None,
    ) -> None:
        """Add a single step to the partial trajectory identified by `key`.

        Generally a single step could correspond to, e.g., one environment managed
        by a VecEnv.

        Args:
            step_dict: dictionary containing information for the current step. Its
                keys could include any (or all) attributes of a `TrajectoryWithRew`
                (e.g. "obs", "acts", etc.).
            key: key to uniquely identify the trajectory to append to, if working
                with multiple partial trajectories.
        """
        self.partial_trajectories[key].append(step_dict)

    def finish_trajectory(
        self,
        key, #: Hashable,
        terminal: bool,
    ):  # -> types.TrajectoryWithRew:
        """Complete the trajectory labelled with `key`.

        Args:
            key: key uniquely identifying which in-progress trajectory to remove.
            terminal: trajectory has naturally finished (i.e. includes terminal state).

        Returns:
            traj: list of completed trajectories popped from
                `self.partial_trajectories`.
        """
        part_dicts = self.partial_trajectories[key]
        del self.partial_trajectories[key]
        out_dict_unstacked = collections.defaultdict(list)
        for part_dict in part_dicts:
            for key, array in part_dict.items():
                out_dict_unstacked[key].append(array)
        out_dict_stacked = {
            key: np.stack(arr_list, axis=0)
            for key, arr_list in out_dict_unstacked.items()
        }
        traj = TrajectoryWithRew(**out_dict_stacked, terminal=terminal)
        assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1
        return traj

    def add_steps_and_auto_finish(
        self,
        acts: np.ndarray,
        obs: np.ndarray,
        rews: np.ndarray,
        dones: np.ndarray,
        infos: List[dict],
    ): #-> List[types.TrajectoryWithRew]:
        """Calls `add_step` repeatedly using acts and the returns from `venv.step`.

        Also automatically calls `finish_trajectory()` for each `done == True`.
        Before calling this method, each environment index key needs to be
        initialized with the initial observation (usually from `venv.reset()`).

        See the body of `util.rollout.generate_trajectory` for an example.

        Args:
            acts: Actions passed into `VecEnv.step()`.
            obs: Return value from `VecEnv.step(acts)`.
            rews: Return value from `VecEnv.step(acts)`.
            dones: Return value from `VecEnv.step(acts)`.
            infos: Return value from `VecEnv.step(acts)`.

        Returns:
            A list of completed trajectories. There should be one trajectory for
            each `True` in the `dones` argument.
        """
        trajs = []
        for env_idx in range(len(obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
                "Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)"
            )

        zip_iter = enumerate(zip(acts, obs, rews, dones, infos))
        for env_idx, (act, ob, rew, done, info) in zip_iter:
            if done:
                # When dones[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv, and
                # infos[i]["terminal_observation"] is the actual final observation.
                real_ob = info["terminal_observation"]
            else:
                real_ob = ob

            self.add_step(
                dict(
                    acts=act,
                    # rews=rew,
                    # this is not the obs corresponding to `act`, but rather the obs
                    # *after* `act` (see above)
                    obs=real_ob,
                    # infos=info,
                ),
                env_idx,
            )
            if done:
                # finish env_idx-th trajectory
                new_traj = self.finish_trajectory(env_idx, terminal=True)
                trajs.append(new_traj)
                # When done[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                self.add_step(dict(obs=ob), env_idx)
        return trajs


def flatten_trajectories(
    trajectories  # : Sequence[Trajectory],
): #-> types.Transitions:
    """Flatten a series of trajectory dictionaries into arrays.

    Args:
        trajectories: list of trajectories.

    Returns:
        The trajectories flattened into a single batch of Transitions.
    """
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts = {key: [] for key in keys}
    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        if traj.infos is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = traj.infos
        parts["infos"].append(infos)

    cat_parts = {
        key: np.concatenate(part_list, axis=0) for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return Transitions(**cat_parts)


def generate_trajectories(
    policy,  #: AnyPolicy,
    venv,  #: VecEnv,
    sample_until,  #: GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
): #-> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    get_actions = _policy_to_callable(policy, venv, deterministic_policy)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    while np.any(active):
        acts = get_actions(obs)
        obs, rews, dones, infos = venv.step(acts)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any environments
            # where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories


def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
):  # -> Mapping[str, Union[np.ndarray, th.Tensor]]:
    """Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.

    Use this as the `collate_fn` argument to `DataLoader` if using an instance of
    `TransitionsMinimal` as the `dataset` argument.

    Args:
        batch: The batch to collate.

    Returns:
        A collated batch. Uses Torch's default collate function for everything
        except the "infos" key. For "infos", we join all the info dicts into a
        list of dicts. (The default behavior would recursively collate every
        info dict into a single dict, which is incorrect.)
    """
    batch_no_infos = [
        {k: np.array(v) for k, v in sample.items() if k != "infos"} for sample in batch
    ]

    result = th_data.dataloader.default_collate(batch_no_infos)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample in batch]
    return result


def make_min_episodes(n: int):  # -> GenTrajTerminationFn:
    """Terminate after collecting n episodes of data.

    Args:
        n: Minimum number of episodes of data to collect.
            May overshoot if two episodes complete simultaneously (unlikely).

    Returns:
        A function implementing this termination condition.
    """
    assert n >= 1
    return lambda trajectories: len(trajectories) >= n


class EpochOrBatchIteratorWithProgress:
    """Wraps DataLoader so that all BC batches can be processed in one for-loop.

    Also uses `tqdm` to show progress in stdout.
    """

    def __init__(
        self,
        data_loader, #: Iterable[algo_base.TransitionMapping],
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        progress_bar_visible: bool = True,
    ):
        """Builds EpochOrBatchIteratorWithProgress.

        Args:
            data_loader: An iterable over data dicts, as used in `BC`.
            n_epochs: The number of epochs to iterate through in one call to
                __iter__. Exactly one of `n_epochs` and `n_batches` should be provided.
            n_batches: The number of batches to iterate through in one call to
                __iter__. Exactly one of `n_epochs` and `n_batches` should be provided.
            on_epoch_end: A callback function without parameters to be called at the
                end of every epoch.
            on_batch_end: A callback function without parameters to be called at the
                end of every batch.
            progress_bar_visible: If True, then show a tqdm progress bar.

        Raises:
            ValueError: If neither or both of `n_epochs` and `n_batches` are non-None.
        """
        if n_epochs is not None and n_batches is None:
            self.use_epochs = True
        elif n_epochs is None and n_batches is not None:
            self.use_epochs = False
        else:
            raise ValueError(
                "Must provide exactly one of `n_epochs` and `n_batches` arguments.",
            )

        self.data_loader = data_loader
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.on_epoch_end = on_epoch_end
        self.on_batch_end = on_batch_end
        self.progress_bar_visible = progress_bar_visible

    def __iter__(
        self,
    ): #-> Iterable[Tuple[algo_base.TransitionMapping, Mapping[str, Any]]]:
        """Yields batches while updating tqdm display to display progress."""
        samples_so_far = 0
        epoch_num = 0
        batch_num = 0
        batch_suffix = epoch_suffix = ""
        if self.progress_bar_visible:
            if self.use_epochs:
                display = tqdm.tqdm(total=self.n_epochs)
                epoch_suffix = f"/{self.n_epochs}"
            else:  # Use batches.
                display = tqdm.tqdm(total=self.n_batches)
                batch_suffix = f"/{self.n_batches}"
        else:
            display = _NoopTqdm()

        def update_desc():
            display.set_description(
                f"batch: {batch_num}{batch_suffix}  epoch: {epoch_num}{epoch_suffix}",
            )

        with contextlib.closing(display):
            while True:
                update_desc()
                got_data_on_epoch = False
                for batch in self.data_loader:
                    got_data_on_epoch = True
                    batch_num += 1
                    batch_size = len(batch["obs"])
                    assert batch_size > 0
                    samples_so_far += batch_size
                    stats = dict(
                        epoch_num=epoch_num,
                        batch_num=batch_num,
                        samples_so_far=samples_so_far,
                    )
                    yield batch, stats
                    if self.on_batch_end is not None:
                        self.on_batch_end()
                    if not self.use_epochs:
                        update_desc()
                        display.update(1)
                        if batch_num >= self.n_batches:
                            return
                if not got_data_on_epoch:
                    raise AssertionError(
                        f"Data loader returned no data after "
                        f"{batch_num} batches, during epoch "
                        f"{epoch_num} -- did it reset correctly?",
                    )
                epoch_num += 1
                if self.on_epoch_end is not None:
                    self.on_epoch_end()

                if self.use_epochs:
                    update_desc()
                    display.update(1)
                    if epoch_num >= self.n_epochs:
                        return


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
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Optional[policies.ActorCriticPolicy] = None,
        demonstrations=None,  # Optional[algo_base.AnyTransitions] = None,
        batch_size: int = 32,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, torch.device] = "auto",
        # custom_logger: Optional[logger.HierarchicalLogger] = None,
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
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: If `weight_decay` is specified in `optimizer_kwargs` (use the
                parameter `l2_weight` instead.)
        """
        self._demo_data_loader = None # Optional[Iterable[TransitionMapping]] = None
        self.batch_size = batch_size
        # super().__init__(
        #     demonstrations=demonstrations,
        #     custom_logger=custom_logger,
        # )

        if demonstrations is not None:
            self.set_demonstrations(demonstrations)

        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        self.tensorboard_step = 0

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
        self._policy = policy.to(self.device)
        # TODO(adam): make policy mandatory and delete observation/action space params?
        assert self.policy.observation_space == self.observation_space
        assert self.policy.action_space == self.action_space

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )

        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

    @property
    def policy(self) -> policies.ActorCriticPolicy:
        return self._policy

    def set_demonstrations(self, demonstrations): # algo_base.AnyTransitions) -> None:
        self._demo_data_loader = self.make_data_loader(
            demonstrations,
            self.batch_size,
        )

    def _calculate_loss(
        self,
        obs: Union[torch.Tensor, np.ndarray],
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

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            on_epoch_end: Optional callback with no parameters to run at the end of each
                epoch.
            on_batch_end: Optional callback with no parameters to run at the end of each
                batch.
            log_interval: Log stats after every log_interval batches.
            log_rollouts_venv: If not None, then this VecEnv (whose observation and
                actions spaces must match `self.observation_space` and
                `self.action_space`) is used to generate rollout stats, including
                average return and average episode length. If None, then no rollouts
                are generated.
            log_rollouts_n_episodes: Number of rollouts to generate when calculating
                rollout stats. Non-positive number disables rollouts.
            progress_bar: If True, then show a progress bar during training.
            reset_tensorboard: If True, then start plotting to Tensorboard from x=0
                even if `.train()` logged to Tensorboard previously. Has no practical
                effect if `.train()` is being called for the first time.
        """
        it = EpochOrBatchIteratorWithProgress(
            self._demo_data_loader,
            n_epochs=n_epochs,
            n_batches=n_batches,
            on_epoch_end=on_epoch_end,
            on_batch_end=on_batch_end,
            progress_bar_visible=progress_bar,
        )

        if reset_tensorboard:
            self.tensorboard_step = 0

        batch_num = 0
        for batch, stats_dict_it in it:
            loss, stats_dict_loss = self._calculate_loss(batch["obs"], batch["acts"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if batch_num % log_interval == 0:
                # for stats in [stats_dict_it, stats_dict_loss]:
                #     for k, v in stats.items():
                #         self.logger.record(f"bc/{k}", v)
                # TODO(shwang): Maybe instead use a callback that can be shared between
                #   all algorithms' `.train()` for generating rollout stats.
                #   EvalCallback could be a good fit:
                #   https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback
                # if log_rollouts_venv is not None and log_rollouts_n_episodes > 0:
                #     trajs = generate_trajectories(
                #         self.policy,
                #         log_rollouts_venv,
                #         make_min_episodes(log_rollouts_n_episodes),
                #     )
                #    stats = rollout_stats(trajs)
                #     self.logger.record("batch_size", len(batch["obs"]))
                #     for k, v in stats.items():
                #         if "return" in k and "monitor" not in k:
                #             self.logger.record("rollout/" + k, v)
                # self.logger.dump(self.tensorboard_step)
            batch_num += 1
            self.tensorboard_step += 1

    def save_policy(self, policy_path):  # : types.AnyPath) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        torch.save(self.policy, policy_path)

    @staticmethod
    def make_data_loader(
            transitions, #: AnyTransitions,
            batch_size: int,
            data_loader_kwargs: Optional[Mapping[str, Any]] = None,
    ): #-> Iterable[TransitionMapping]:
        """Converts demonstration data to Torch data loader.

        Args:
            transitions: Transitions expressed directly as a `types.TransitionsMinimal`
                object, a sequence of trajectories, or an iterable of transition
                batches (mappings from keywords to arrays containing observations, etc).
            batch_size: The size of the batch to create. Does not change the batch size
                if `transitions` is already an iterable of transition batches.
            data_loader_kwargs: Arguments to pass to `th_data.DataLoader`.

        Returns:
            An iterable of transition batches.

        Raises:
            ValueError: if `transitions` is an iterable over transition batches with batch
                size not equal to `batch_size`; or if `transitions` is transitions or a
                sequence of trajectories with total timesteps less than `batch_size`.
            TypeError: if `transitions` is an unsupported type.
        """

        if batch_size <= 0:
            raise ValueError(f"batch_size={batch_size} must be positive.")

        if isinstance(transitions, Iterable):
            try:
                first_item = next(iter(transitions))
            except StopIteration:
                first_item = None
            if isinstance(first_item, Trajectory):
                transitions = flatten_trajectories(list(transitions))
        # from imitation.data import types
        if isinstance(transitions, TransitionsMinimal):  # or isinstance(transitions, types.TransitionsMinimal):
            if len(transitions) < batch_size:
                raise ValueError(
                    f"Number of transitions in `demonstrations` {len(transitions)} "
                    f"is smaller than batch size {batch_size}.",
                )

            extra_kwargs = dict(shuffle=True, drop_last=True)
            if data_loader_kwargs is not None:
                extra_kwargs.update(data_loader_kwargs)
            return th_data.DataLoader(
                transitions,
                batch_size=batch_size,
                collate_fn=transitions_collate_fn,
                **extra_kwargs,
            )
        elif isinstance(transitions, Iterable):
            return _WrappedDataLoader(transitions, batch_size)
        else:
            raise TypeError(f"`demonstrations` unexpected type {type(transitions)}")

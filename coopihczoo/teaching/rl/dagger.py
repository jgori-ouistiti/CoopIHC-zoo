import pathlib
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union
from stable_baselines3.common import policies, utils, vec_env
import numpy as np


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


def _save_dagger_demo(
    trajectory,  # : types.Trajectory,
    save_dir,    # : types.AnyPath,
    prefix: str = "",
) -> None:
    # TODO(shwang): This is possibly redundant with types.save(). Note
    #   however that NPZ save here is likely more space efficient than
    #   pickle from types.save(), and types.save only accepts
    #   TrajectoryWithRew right now (subclass of Trajectory).
    save_dir = pathlib.Path(save_dir)
    # assert isinstance(trajectory, types.Trajectory)
    actual_prefix = f"{prefix}-" if prefix else ""
    timestamp = util.make_unique_timestamp()
    filename = f"{actual_prefix}dagger-demo-{timestamp}.npz"

    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = pathlib.Path(save_dir, filename)
    np.savez_compressed(npz_path, **dataclasses.asdict(trajectory))
    # logging.info(f"Saved demo at '{npz_path}'")


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



class SimpleDAggerTrainer:
    """Simpler subclass of DAggerTrainer for training with synthetic feedback."""

    def __init__(
        self,
        *,
        venv,           #: vec_env.VecEnv,
        scratch_dir,    #: types.AnyPath,
        expert_policy,  #: policies.BasePolicy,
        expert_trajs=None   #: Optional[Sequence[types.Trajectory]] = None,
    ):
        """Builds SimpleDAggerTrainer.

        Args:
            venv: Vectorized training environment. Note that when the robot
                action is randomly injected (in accordance with `beta_schedule`
                argument), every individual environment will get a robot action
                simultaneously for that timestep.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            expert_policy: The expert policy used to generate synthetic demonstrations.
            expert_trajs: Optional starting dataset that is inserted into the round 0
                dataset.
            dagger_trainer_kwargs: Other keyword arguments passed to the
                superclass initializer `DAggerTrainer.__init__`.

        Raises:
            ValueError: The observation or action space does not match between
                `venv` and `expert_policy`.
        """
        self.venv = venv
        self.expert_policy = expert_policy
        if expert_policy.observation_space != self.venv.observation_space:
            raise ValueError(
                "Mismatched observation space between expert_policy and venv",
            )
        if expert_policy.action_space != self.venv.action_space:
            raise ValueError("Mismatched action space between expert_policy and venv")

        # TODO(shwang):
        #   Might welcome Transitions and DataLoaders as sources of expert data
        #   in the future too, but this will require some refactoring, so for
        #   now we just have `expert_trajs`.
        if expert_trajs is not None:
            # Save each initial expert trajectory into the "round 0" demonstration
            # data directory.
            for traj in expert_trajs:
                _save_dagger_demo(
                    traj,
                    self._demo_dir_path_for_round(),
                    prefix="initial_data",
                )

    def train(
        self,
        total_timesteps: int,
        *,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs=None #Optional[dict] = None,
    ) -> None:
        """Train the DAgger agent.

        The agent is trained in "rounds" where each round consists of a dataset
        aggregation step followed by BC update step.

        During a dataset aggregation step, `self.expert_policy` is used to perform
        rollouts in the environment but there is a `1 - beta` chance (beta is
        determined from the round number and `self.beta_schedule`) that the DAgger
        agent's action is used instead. Regardless of whether the DAgger agent's action
        is used during the rollout, the expert action and corresponding observation are
        always appended to the dataset. The number of environment steps in the
        dataset aggregation stage is determined by the `rollout_round_min*` arguments.

        During a BC update step, `BC.train()` is called to update the DAgger agent on
        all data collected so far.

        Args:
            total_timesteps: The number of timesteps to train inside the environment.
                In practice this is a lower bound, because the number of timesteps is
                rounded up to finish the minimum number of episdoes or timesteps in the
                last DAgger training round, and the environment timesteps are executed
                in multiples of `self.venv.num_envs`.
            rollout_round_min_episodes: The number of episodes the must be completed
                completed before a dataset aggregation step ends.
            rollout_round_min_timesteps: The number of environment timesteps that must
                be completed before a dataset aggregation step ends. Also, that any
                round will always train for at least `self.batch_size` timesteps,
                because otherwise BC could fail to receive any batches.
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.
        """
        total_timestep_count = 0
        round_num = 0

        while total_timestep_count < total_timesteps:
            collector = self.create_trajectory_collector()
            round_episode_count = 0
            round_timestep_count = 0

            sample_until = rollout.make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            trajectories = rollout.generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=collector.rng,
            )

            for traj in trajectories:
                self._logger.record_mean(
                    "dagger/mean_episode_reward",
                    np.sum(traj.rews),
                )
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)

            round_episode_count += len(trajectories)

            # `logger.dump` is called inside BC.train within the following fn call:
            self.extend_and_update(bc_train_kwargs)
            round_num += 1

    def extend_and_update(self, bc_train_kwargs: Optional[Mapping] = None) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.create_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.

        Returns:
            New round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()
        if "log_rollouts_venv" not in user_keys:
            bc_train_kwargs["log_rollouts_venv"] = self.venv

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        logging.info("Loading demonstrations")
        self._try_load_demos()
        logging.info(f"Training at round {self.round_num}")
        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        logging.info(f"New round number is {self.round_num}")
        return self.round_num

    def _demo_dir_path_for_round(self, round_num: Optional[int] = None) -> pathlib.Path:
        if round_num is None:
            round_num = self.round_num
        return self.scratch_dir / "demos" / f"round-{round_num:03d}"

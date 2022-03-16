import pathlib
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union
import dataclasses
import datetime
import uuid
import os
import abc

from stable_baselines3.common import vec_env
import numpy as np
import torch.utils.data as th_data

from .behavioral_cloning import Trajectory, generate_trajectories, make_min_episodes, \
    TrajectoryAccumulator, TrajectoryWithRew, Transitions, transitions_collate_fn


class NeedsDemosException(Exception):
    """Signals demos need to be collected for current round before continuing."""


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_uuid}"


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
    timestamp = make_unique_timestamp()
    filename = f"{actual_prefix}dagger-demo-{timestamp}.npz"

    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = pathlib.Path(save_dir, filename)
    np.savez_compressed(npz_path, **dataclasses.asdict(trajectory))
    # logging.info(f"Saved demo at '{npz_path}'")


def _load_trajectory(npz_path: str) -> Trajectory:
    """Load a single trajectory from a compressed Numpy file."""
    np_data = np.load(npz_path, allow_pickle=True)
    has_rew = "rews" in np_data
    dict_data = dict(np_data.items())

    # infos=None is saved as array(None) which leads to a type checking error upon
    # `Trajectory` initialization. Convert to None to prevent error.
    infos = dict_data["infos"]
    if infos.shape == ():
        assert infos.item() is None
        dict_data["infos"] = None

    cls = TrajectoryWithRew if has_rew else Trajectory
    return cls(**dict_data)


class InteractiveTrajectoryCollector(vec_env.VecEnvWrapper):
    """DAgger VecEnvWrapper for querying and saving expert actions.

    Every call to `.step(actions)` accepts and saves expert actions to `self.save_dir`,
    but only forwards expert actions to the wrapped VecEnv with probability
    `self.beta`. With probability `1 - self.beta`, a "robot" action (i.e
    an action from the imitation policy) is forwarded instead.

    Demonstrations are saved as `TrajectoryWithRew` to `self.save_dir` at the end
    of every episode.
    """

    def __init__(
        self,
        venv: vec_env.VecEnv,
        get_robot_acts: Callable[[np.ndarray], np.ndarray],
        beta: float,
        save_dir  # : types.AnyPath,
    ):
        """Builds InteractiveTrajectoryCollector.

        Args:
            venv: vectorized environment to sample trajectories from.
            get_robot_acts: get robot actions that can be substituted for
                human actions. Takes a vector of observations as input & returns a
                vector of actions.
            beta: fraction of the time to use action given to .step() instead of
                robot action. The choice of robot or human action is independently
                randomized for each individual `Env` at every timestep.
            save_dir: directory to save collected trajectories in.
        """
        super().__init__(venv)
        self.get_robot_acts = get_robot_acts
        assert 0 <= beta <= 1
        self.beta = beta
        self.traj_accum = None
        self.save_dir = save_dir
        self._last_obs = None
        self._done_before = True
        self._is_reset = False
        self._last_user_actions = None
        self.rng = np.random.RandomState()

    def seed(self, seed=Optional[int]) -> List[Union[None, int]]:
        """Set the seed for the DAgger random number generator and wrapped VecEnv.

        The DAgger RNG is used along with `self.beta` to determine whether the expert
        or robot action is forwarded to the wrapped VecEnv.

        Args:
            seed: The random seed. May be None for completely random seeding.

        Returns:
            A list containing the seeds for each individual env. Note that all list
            elements may be None, if the env does not return anything when seeded.
        """
        self.rng = np.random.RandomState(seed=seed)
        return self.venv.seed(seed)

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            obs: first observation of a new trajectory.
        """
        self.traj_accum = TrajectoryAccumulator()
        obs = self.venv.reset()
        for i, ob in enumerate(obs):
            self.traj_accum.add_step({"obs": ob}, key=i)
        self._last_obs = obs
        self._is_reset = True
        self._last_user_actions = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Steps with a `1 - beta` chance of using `self.get_robot_acts` instead.

        DAgger needs to be able to inject imitation policy actions randomly at some
        subset of time steps. This method has a `self.beta` chance of keeping the
        `actions` passed in as an argument, and a `1 - self.beta` chance of
        forwarding actions generated by `self.get_robot_acts` instead.
        "robot" (i.e. imitation policy) action if necessary.

        At the end of every episode, a `TrajectoryWithRew` is saved to `self.save_dir`,
        where every saved action is the expert action, regardless of whether the
        robot action was used during that timestep.

        Args:
            actions: the _intended_ demonstrator/expert actions for the current
                state. This will be executed with probability `self.beta`.
                Otherwise, a "robot" (typically a BC policy) action will be sampled
                and executed instead via `self.get_robot_act`.
        """
        assert self._is_reset, "call .reset() before .step()"

        # Replace each given action with a robot action 100*(1-beta)% of the time.
        actual_acts = np.array(actions)

        mask = self.rng.uniform(0, 1, size=(self.num_envs,)) > self.beta
        if np.sum(mask) != 0:
            actual_acts[mask] = self.get_robot_acts(self._last_obs[mask])

        self._last_user_actions = actions
        self.venv.step_async(actual_acts)

    def step_wait(self): #-> VecEnvStepReturn:
        """Returns observation, reward, etc after previous `step_async()` call.

        Stores the transition, and saves trajectory as demo once complete.

        Returns:
            Observation, reward, dones (is terminal?) and info dict.
        """
        next_obs, rews, dones, infos = self.venv.step_wait()
        self._last_obs = next_obs
        fresh_demos = self.traj_accum.add_steps_and_auto_finish(
            obs=next_obs,
            acts=self._last_user_actions,
            rews=rews,
            infos=infos,
            dones=dones,
        )
        for traj in fresh_demos:
            _save_dagger_demo(traj, self.save_dir)

        return next_obs, rews, dones, infos


class BetaSchedule(abc.ABC):
    """Computes beta (% of time demonstration action used) from training round."""

    @abc.abstractmethod
    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round.

        Args:
            round_num: the current round number. Rounds are assumed to be sequentially
                numbered from 0.

        Returns:
            The fraction of the time to sample a demonstrator action. Robot
                actions will be sampled the remainder of the time.
        """  # noqa: DAR202


class LinearBetaSchedule(BetaSchedule):
    """Linearly-decreasing schedule for beta."""

    def __init__(self, rampdown_rounds: int):
        """Builds LinearBetaSchedule.

        Args:
            rampdown_rounds: number of rounds over which to anneal beta.
        """
        self.rampdown_rounds = rampdown_rounds

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta linearly decreasing from `1` to `0` between round `0` and
            `self.rampdown_rounds`. After that, it is 0.
        """
        assert round_num >= 0
        return min(1, max(0, (self.rampdown_rounds - round_num) / self.rampdown_rounds))


class SimpleDAggerTrainer:
    """Simpler subclass of DAggerTrainer for training with synthetic feedback."""

    DEFAULT_N_EPOCHS: int = 4
    """The default number of BC training epochs in `extend_and_update`."""

    def __init__(
        self,
        *,
        venv,           #: vec_env.VecEnv,
        scratch_dir,    #: types.AnyPath,
        expert_policy,  #: policies.BasePolicy,
        bc_trainer,
        expert_trajs=None,  #: Optional[Sequence[types.Trajectory]] = None,
        beta_schedule=None
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
        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)
        self.beta_schedule = beta_schedule

        self.bc_trainer = bc_trainer

        self.scratch_dir = pathlib.Path(scratch_dir)

        self.round_num = 0
        self._last_loaded_round = -1
        self._all_demos = []

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

            sample_until = make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            trajectories = generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=collector.rng,
            )

            for traj in trajectories:
                # self._logger.record_mean(
                #     "dagger/mean_episode_reward",
                #     np.sum(traj.rews),
                # )
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

        # logging.info("Loading demonstrations")
        self._try_load_demos()
        # logging.info(f"Training at round {self.round_num}")
        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        # logging.info(f"New round number is {self.round_num}")
        return self.round_num

    def _demo_dir_path_for_round(self, round_num: Optional[int] = None) -> pathlib.Path:
        if round_num is None:
            round_num = self.round_num
        return self.scratch_dir / "demos" / f"round-{round_num:03d}"

    def create_trajectory_collector(self): #-> InteractiveTrajectoryCollector:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """
        save_dir = self._demo_dir_path_for_round()
        beta = self.beta_schedule(self.round_num)
        collector = InteractiveTrajectoryCollector(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            beta=beta,
            save_dir=save_dir,
        )
        return collector

    @property
    def batch_size(self) -> int:
        return self.bc_trainer.batch_size

    def _try_load_demos(self) -> None:
        """Load the dataset for this round into self.bc_trainer as a DataLoader."""
        demo_dir = self._demo_dir_path_for_round()
        demo_paths = self._get_demo_paths(demo_dir) if os.path.isdir(demo_dir) else []
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".create_trajectory_collector()",
            )

        if self._last_loaded_round < self.round_num:
            transitions, num_demos = self._load_all_demos()
            # logging.info(
            #     f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds",
            # )
            if len(transitions) < self.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.batch_size={self.batch_size} > "
                    f"len(transitions)={len(transitions)}",
                )
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=transitions_collate_fn,
            )
            self.bc_trainer.set_demonstrations(data_loader)
            self._last_loaded_round = self.round_num

    def _get_demo_paths(self, round_dir):
        return [
            os.path.join(round_dir, p)
            for p in os.listdir(round_dir)
            if p.endswith(".npz")
        ]

    def _load_all_demos(self):
        num_demos_by_round = []
        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)
            self._all_demos.extend(_load_trajectory(p) for p in demo_paths)
            num_demos_by_round.append(len(demo_paths))
        # logging.info(f"Loaded {len(self._all_demos)} total")
        demo_transitions = flatten_trajectories(self._all_demos)
        return demo_transitions, num_demos_by_round

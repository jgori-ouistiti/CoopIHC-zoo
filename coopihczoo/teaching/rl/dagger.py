import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import coopihc


class LinearBetaSchedule:
    """Computes beta (% of time demonstration action used) from training round.
    Linearly-decreasing schedule for beta."""

    def __init__(self, rampdown_rounds: int):
        """Builds LinearBetaSchedule.

        Args:
            rampdown_rounds: number of rounds over which to anneal beta.
        """
        self.rampdown_rounds = rampdown_rounds

    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round.

        Args:
            round_num: the current round number. Rounds are assumed to be sequentially
                numbered from 0.

        Returns:
             The fraction of the time to sample a demonstrator action. Robot
                actions will be sampled the remainder of the time.
            beta linearly decreasing from `1` to `0` between round `0` and
            `self.rampdown_rounds`. After that, it is 0.
        """
        assert round_num >= 0
        return min(1, max(0, (self.rampdown_rounds - round_num) / self.rampdown_rounds))


class DAgger:

    def __init__(
        self,
        env,
        expert,
        policy,
        batch_size,
        expert_trajs=None,
        beta_schedule=None
    ):
        self.env = env
        self.expert = expert
        # if expert.observation_space != self.env.observation_space:
        #     raise ValueError(
        #         "Mismatched observation space between expert_policy and venv",
        #     )
        # if expert.action_space != self.env.action_space:
        #     raise ValueError("Mismatched action space between expert_policy and venv")

        if expert_trajs is None:
            expert_trajs = []

        self.expert_trajs = expert_trajs

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)

        self.beta_schedule = beta_schedule

        self.bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            policy=policy,
            batch_size=total_n_iter)

        self.round_num = 0
        self._last_loaded_round = -1

    def train(
        self,
        total_timesteps: int,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
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

        while total_timestep_count < total_timesteps:

            trajectories = self.sample_expert(
                env=self.env, expert=self.expert,
                n_timesteps=max(rollout_round_min_timesteps, self.bc_trainer.batch_size),
                n_episode=rollout_round_min_episodes)

            self.expert_trajs.extend(trajectories)

            self.bc_trainer.set_demonstrations(self.expert_trajs)
            self.bc_trainer.train()

            total_timestep_count += len(trajectories)
            self.round_num += 1

    def sample_expert(self, env, expert, n_episode=50, n_timesteps=None, deterministic=True):

        beta = self.beta_schedule(self.round_num)

        # env = VecMonitor(DummyVecEnv([lambda: env]))

        obs = env.reset()

        n_steps = 0
        ep = 0

        expert_data = [[], ]

        with torch.no_grad():
            while True:

                if np.random.random() > beta:
                    # Student takes the decision
                    # expert_chose = False
                    action, _states = self.bc_trainer.policy.predict(obs, deterministic=deterministic)

                else:
                    # Expert takes the decision
                    # expert_chose = True
                    if isinstance(expert, coopihc.BaseAgent):
                        action, _reward = env.unwrapped.bundle.assistant._take_action()
                        action = int(action)
                    else:
                        action, _state = expert.predict(obs, deterministic=deterministic)

                new_obs, reward, done, info = env.step(action)

                n_steps += 1

                # action = action.squeeze()

                # if isinstance(obs, torch.Tensor) or isinstance(obs, np.ndarray):
                #     obs = obs.squeeze()
                # else:
                #     for k, v in obs.items():
                #         if len(v.squeeze().shape):
                #             obs[k] = v.squeeze()
                #         else:
                #             obs[k] = v.squeeze(-1).squeeze(-1)
                            # print(v.squeeze(-1).squeeze(-1).shape)

                # if expert_chose:
                expert_data[-1].append({"acts": action, "obs": obs})

                # Handle timeout by bootstraping with value function
                # see GitHub issue #633

                if done:
                    env.reset()

                    ep += 1
                    expert_data.append([])

                obs = new_obs

                if n_episode is not None and ep < n_episode:
                    continue

                if n_timesteps is not None and n_steps < n_timesteps:
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

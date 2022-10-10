from abc import ABC
import torch
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from coopihc import BaseAgent, State, \
    cat_element, array_element, \
    RuleObservationEngine, oracle_engine_specification

from . conservative_sampling import ConservativeSamplingPolicy
from . rl import RlTeacherInferenceEngine
from stable_baselines3.common.policies import BasePolicy


class ConservativeSamplingExpert(BaseAgent, BasePolicy, ABC):

    def __init__(self, *args, **kwargs):
        super(ConservativeSamplingExpert, self).__init__("assistant", *args, **kwargs)

    def finit(self, *args, **kwargs):

        n_item = int(self.bundle.task.state.n_item)
        n_session = int(self.bundle.task.state.n_session)
        inter_trial = int(self.bundle.task.state.inter_trial)
        n_iter_per_ss = int(self.bundle.task.state.n_iter_per_ss)
        break_length = int(self.bundle.task.state.break_length)
        log_thr = float(self.bundle.task.state.log_thr)
        is_item_specific = bool(self.bundle.task.state.is_item_specific)

        self.state["progress"] = array_element(low=0, high=np.inf, init=np.zeros(1))
        self.state["memory"] = array_element(low=0, high=np.inf, init=np.zeros((n_item, 2)))

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)

        agent_policy = ConservativeSamplingPolicy(
            action_state=action_state,
            n_session=n_session,
            inter_trial=inter_trial,
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            break_length=break_length,
            log_thr=log_thr,
            is_item_specific=is_item_specific)

        # Inference engine
        inference_engine = RlTeacherInferenceEngine()  # This is specific!!!

        # Use default observation engine
        observation_engine = RuleObservationEngine(
            deterministic_specification=oracle_engine_specification)

        self._attach_policy(agent_policy)
        self._attach_observation_engine(observation_engine)
        self._attach_inference_engine(inference_engine)

    def reset(self, dic=None):

        n_item = int(self.bundle.task.state["n_item"])

        self.state["progress"] = 0
        self.state["memory"] = np.zeros((n_item, 2))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        obs = observation

        if state is not None or episode_start is not None:
            raise NotImplementedError

        if self.policy.is_item_specific:
            init_forget_rate = obs["user_state"]["param"][:, 0]
            rep_effect = obs["user_state"]["param"][:, 1]

        else:
            init_forget_rate, rep_effect = obs["param"].squeeze()

        iteration = obs["iteration"].squeeze()
        session = obs["session"].squeeze()
        n_pres = obs["n_pres"].squeeze()
        last_pres = obs["last_pres"].squeeze()
        timestamp = obs["timestamp"].squeeze()

        item = self.policy._loop(
            timestamp=timestamp,
            iteration=iteration,
            session=session,
            n_pres=n_pres,
            last_pres=last_pres,
            init_forget_rate=init_forget_rate,
            rep_effect=rep_effect)

        return np.array([item, ]), None

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

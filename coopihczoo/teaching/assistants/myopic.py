from coopihc import (
    BaseAgent,
    State,
    cat_element,
    BasePolicy,
    RuleObservationEngine,
    oracle_engine_specification
)
import numpy as np


class MyopicPolicy(BasePolicy):

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    def _threshold_select(
            self, n_pres, initial_forget_rates,
            initial_repetition_rates, n_item,
            delta,
            log_thr):

        if np.max(n_pres) == 0:
            item = 0
        else:
            seen = n_pres > 0

            log_p_seen = self._cp_log_p_seen(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=initial_forget_rates,
                initial_repetition_rates=initial_repetition_rates)

            if np.sum(seen) == n_item \
                    or np.min(log_p_seen) <= log_thr:

                item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
            else:
                item = np.argmin(seen)

        return item

    def _cp_log_p_seen(
            self,
            n_pres,
            delta,
            initial_forget_rates,
            initial_repetition_rates):

        view = n_pres > 0
        rep = n_pres[view] - 1.
        delta = delta[view]

        if self.observation.task_state.is_item_specific:

            init_forget = initial_forget_rates[np.nonzero(view)]
            rep_effect = initial_repetition_rates[np.nonzero(view)]

        else:
            init_forget = initial_forget_rates
            rep_effect = initial_repetition_rates

        forget_rate = init_forget * (1 - rep_effect) ** rep
        logp_recall = - forget_rate * delta
        return logp_recall

    def sample(self, observation=None, **kwargs):

        is_item_specific = bool(self.observation.task_state.is_item_specific)

        if is_item_specific:
            init_forget_rate = self.observation["user_state"]["param"][:, 0]
            rep_effect = self.observation["user_state"]["param"][:, 1]

        else:
            init_forget_rate = self.observation["user_state"]["param"][0]
            rep_effect = self.observation["user_state"]["param"][1]

        n_item = int(self.observation.task_state.n_item)

        log_thr = float(self.observation.task_state.log_thr)

        delta = int(self.observation.task_state.timestamp) \
            - self.observation.user_state.last_pres.view(np.ndarray)
        n_pres = self.observation.user_state.n_pres.view(np.ndarray)

        _action_value = self._threshold_select(
            n_pres=n_pres, delta=delta,
            initial_forget_rates=init_forget_rate,
            initial_repetition_rates=rep_effect,
            n_item=n_item,
            log_thr=log_thr)

        reward = 0
        return _action_value, reward

    def reset(self, random=True):

        self.action_state["action"] = 0


class Myopic(BaseAgent):

    def __init__(self, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)

    def finit(self, *args, **kwargs):

        n_item = int(self.bundle.task.state["n_item"])

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)

        agent_policy = MyopicPolicy(action_state=action_state)

        # # Inference engine
        # inference_engine = MyopicInferenceEngine()

        observation_engine = RuleObservationEngine(
            deterministic_specification=oracle_engine_specification)

        self._attach_policy(agent_policy)
        self._attach_observation_engine(observation_engine)
        # self.attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset.

        :meta public:
        """
        pass

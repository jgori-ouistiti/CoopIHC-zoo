from coopihc import (
    BaseAgent,
    State,
    cat_element,
    BasePolicy,
)
import numpy as np

from coopihczoo.teaching.memory_models import ExponentialDecayMemory


class MyopicPolicy(BasePolicy):
    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    def _threshold_select(
        self,
        n_pres,
        delta,
    ):

        if np.max(n_pres) == 0:  # First item
            item = 0
        else:
            n_item = self.parameters["n_item"]
            log_thr = self.parameters["log_thr"]
            seen = n_pres > 0  # compute the probabilities of recall for seen items

            log_p_seen = ExponentialDecayMemory.decay(
                delta_time=delta[seen],
                times_presented=n_pres[seen],
                initial_forgetting_rate=self.retention_params[np.nonzero(seen), 0],
                repetition_effect=self.retention_params[np.nonzero(seen), 1],
                log=True,
            )

            if (
                np.sum(seen) == n_item or np.min(log_p_seen) <= log_thr
            ):  # if we have seen all items or one of the probabilities of recall of a seen item is below the threshold, then select the item with lowest probability of recall as next item

                item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
            else:  # introduce a new item only if all items presented until now have a probability of recall above the threshold
                item = np.argmin(seen)

        return item

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):

        delta = (
            agent_observation.task_state.timestamp
            - agent_observation.task_state.last_pres
        )
        n_pres = agent_observation.task_state.n_pres

        _action_value = self._threshold_select(
            n_pres=n_pres,
            delta=delta,
        )

        reward = 0
        return _action_value, reward

    # def reset(self, random=True):
    #     self.action_state["action"] = 0


class Myopic(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)

    def finit(self, *args, **kwargs):

        n_item = self.parameters["n_item"]

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)

        agent_policy = MyopicPolicy(action_state=action_state)
        self._attach_policy(agent_policy)

from coopihc import (
    BaseAgent,
    State,
    cat_element,
    BasePolicy,
    BaseInferenceEngine,
    discrete_array_element,
)
import numpy as np


class LeitnerInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):
        last_item = int(agent_observation["task_state"]["item"])
        last_time_reply = int(agent_observation["task_state"]["timestamp"])
        last_was_success = bool(agent_observation["user_action"]["action"])

        if agent_observation["game_info"]["round_index"] > 0:

            if last_was_success:
                self.state["box"][last_item] += 1
            else:
                self.state["box"][last_item] = max(0, self.state["box"][last_item] - 1)

            delay = self.delay_factor ** self.state["box"][last_item]
            # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
            self.state["due"][last_item] = last_time_reply + self.delay_min * delay

        reward = 0

        return self.state, reward


class LeitnerPolicy(BasePolicy):
    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):

        box = self.state["box"].flatten()
        due = self.state["due"].flatten()

        n_item = self.parameters["n_item"]

        now = agent_observation["task_state"]["timestamp"]

        if agent_observation["game_info"]["round_index"] == 0:
            _action_value = box.argmin()  # pickup new

        else:
            seen = np.argwhere(box >= 0).flatten()
            n_seen = len(seen)

            if n_seen == n_item:
                _action_value = np.argmin(due)
            else:
                seen__due = due[seen]
                seen__is_due = seen__due <= now
                if np.sum(seen__is_due):
                    seen_and_is_due__due = seen__due[seen__is_due]
                    _action_value = seen[seen__is_due][np.argmin(seen_and_is_due__due)]
                else:
                    _action_value = box.argmin()  # pickup new

        reward = 0

        return _action_value, reward

    def reset(self, random=True):
        self.action_state["action"] = 0


class Leitner(BaseAgent):
    def __init__(self, delay_factor, delay_min, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)
        self.parameters = {"delay_factor": delay_factor, "delay_min": delay_min}

    def finit(self):

        n_item = self.parameters["n_item"]

        self.state["box"] = discrete_array_element(
            low=0, high=np.inf, init=np.zeros(n_item)
        )
        self.state["due"] = discrete_array_element(
            low=0, high=np.inf, init=np.zeros(n_item)
        )

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)
        agent_policy = LeitnerPolicy(action_state)

        # Inference engine
        inference_engine = LeitnerInferenceEngine()

        self._attach_policy(agent_policy)
        self._attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset.

        :meta public:
        """

        n_item = self.parameters["n_item"]

        self.state["box"] = np.zeros(n_item)
        self.state["due"] = np.zeros(n_item)

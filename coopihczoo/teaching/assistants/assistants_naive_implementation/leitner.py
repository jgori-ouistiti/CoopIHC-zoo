import numpy as np
from coopihc import (
    BaseAgent,
    State,
    cat_element,
    BasePolicy,
    BaseInferenceEngine,
    discrete_array_element
)


class LeitnerInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, agent_observation=None):

        last_item = int(self.observation["task_state"]["item"])
        last_time_reply = int(self.observation["task_state"]["timestamp"])
        last_was_success = int(self.observation["user_action"]["action"])

        if self.observation["task_state"]["iteration"] > 0:

            if last_was_success:
                self.state["box"][last_item] += 1
            else:
                self.state["box"][last_item] = max(
                    0, self.state["box"][last_item] - 1
                )

            delay = self.host.delay_factor ** self.state["box"][last_item]
            # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
            self.state["due"][last_item] = (
                last_time_reply + self.host.delay_min * delay
            )

        reward = 0

        return self.state, reward


class LeitnerPolicy(BasePolicy):
    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):

        box = self.state["box"]
        due = self.state["due"]

        n_item = int(self.host.bundle.task.state["n_item"])

        now = self.observation["task_state"]["timestamp"]

        if self.observation["task_state"]["iteration"] == 0:

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

        self.delay_factor = delay_factor
        self.delay_min = delay_min

        super().__init__("assistant", *args, **kwargs)

    def finit(self):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["box"] = discrete_array_element(shape=(n_item, ), low=0, high=np.inf, init=0)
        self.state["due"] = discrete_array_element(shape=(n_item, ), low=0, high=np.inf, init=0)

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)
        agent_policy = LeitnerPolicy(action_state)

        # Inference engine
        inference_engine = LeitnerInferenceEngine()

        # Use default observation engine
        observation_engine = None

        self._attach_policy(agent_policy)
        self._attach_observation_engine(observation_engine)
        self._attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset.

        :meta public:
        """

        n_item = int(self.bundle.task.state["n_item"])

        self.state["box"] = np.zeros(n_item)
        self.state["due"] = np.zeros(n_item)

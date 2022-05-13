from coopihc import (
    BaseAgent,
    State,
    cat_element,
    BasePolicy,
    BaseInferenceEngine,
    array_element,
)
import numpy as np


class LeitnerInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):

        last_item = int(self.observation["task_state"]["item"])
        last_time_reply = self.observation["task_state"]["timestamp"]
        last_was_success = self.observation["user_action"]["action"][0]

        if self.observation["task_state"]["iteration"] > 0:

            if last_was_success:
                self.state["box"][last_item, 0] += 1
            else:
                self.state["box"][last_item, 0] = max(
                    0, self.state["box"][last_item, 0] - 1
                )

            delay = self.host.delay_factor ** self.state["box"][last_item, 0].view(
                np.ndarray
            )
            # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
            self.state["due"][last_item, 0] = (
                last_time_reply + self.host.delay_min * delay
            )

        reward = 0

        return self.state, reward


class LeitnerPolicy(BasePolicy):
    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    def sample(self, observation=None):

        box = self.state["box"].view(np.ndarray).flatten()
        due = self.state["due"].view(np.ndarray).flatten()

        n_item = int(self.host.bundle.task.state["n_item"][0, 0])

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
        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, reward

    def reset(self, random=True):

        _action_value = 0
        self.action_state["action"][:] = _action_value


class Leitner(BaseAgent):
    def __init__(self, delay_factor, delay_min, *args, **kwargs):

        self.delay_factor = delay_factor
        self.delay_min = delay_min

        super().__init__("assistant", *args, **kwargs)

    def finit(self):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["box"] = array_element(shape=n_item, low=0, high=np.inf, init=0)
        self.state["due"] = array_element(shape=n_item, low=0, high=np.inf, init=0)

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(n=n_item)
        agent_policy = LeitnerPolicy(action_state)

        # Inference engine
        inference_engine = LeitnerInferenceEngine()

        # Use default observation engine
        observation_engine = None

        self.attach_policy(agent_policy)
        self.attach_observation_engine(observation_engine)
        self.attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset.

        :meta public:
        """

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["box"][:] = np.zeros(n_item)
        self.state["due"][:] = np.zeros(n_item)

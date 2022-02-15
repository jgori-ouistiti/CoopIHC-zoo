from coopihc import BaseAgent, State, StateElement, Space, ExamplePolicy, \
    BasePolicy, autospace, BaseInferenceEngine
import numpy as np


class AssistantInferenceEngine(BaseInferenceEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):

        last_item = int(self.observation["task_state"]["item"])
        last_time_reply = self.observation["task_state"]["timestamp"]
        last_was_success = self.observation["user_action"]["action"][0]
        # print(type(last_was_success))

        if self.observation["task_state"]["iteration"] > 0:
        
            if last_was_success:
                self.state["box"][last_item, 0] += 1
            else:
                self.state["box"][last_item, 0] = \
                    max(0, self.state["box"][last_item, 0] - 1)

            delay = self.host.delay_factor \
                    ** self.state["box"][last_item, 0].view(np.ndarray)
            # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
            self.state["due"][last_item, 0] = \
                last_time_reply + self.host.delay_min * delay

        reward = 0

        return self.state, reward


class AssistantPolicy(BasePolicy):

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    def sample(self, observation=None):
        """sample

        Compares 'x' to goal and issues +-1 accordingly.

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """

        box = self.state["box"].view(np.ndarray).flatten()
        due = self.state["due"].view(np.ndarray).flatten()

        now = self.observation["task_state"]["timestamp"]

        if self.observation["task_state"]["iteration"] == 0:

            _action_value = box.argmin() # pickup new

        else:

            seen = np.argwhere(box >= 0).flatten()
            n_seen = len(seen)

            if n_seen == self.host.n_item:
                _action_value = np.argmin(due)

            else:
                seen__due = np.asarray(due)[seen]
                seen__is_due = np.asarray(seen__due) <= now
                if np.sum(seen__is_due):
                    seen_and_is_due__due = seen__due[seen__is_due]

                    _action_value  = seen[seen__is_due][np.argmin(seen_and_is_due__due)]
                else:
                    _action_value = box.argmin()  # pickup new

        reward = 0
        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, reward

    def reset(self):

        _action_value = -1
        self.action_state["action"][:] = _action_value


class Assistant(BaseAgent):

    def __init__(self, n_item, delay_factor, delay_min,
                 *args, **kwargs):

        # # Define an internal state with a 'goal' substate
        agent_state = State()
        container = np.atleast_2d(np.zeros(n_item))
        agent_state["box"] = StateElement(
            np.zeros_like(container),
            autospace(
                np.zeros_like(container),
                np.full(container.shape, np.inf),
                dtype=np.float32,
            ),
        )

        agent_state["due"] = StateElement(
            np.zeros_like(container),
            autospace(
                np.zeros_like(container),
                np.full(container.shape, np.inf),
                dtype=np.float32,
            ),
        )

        self.n_item = n_item

        self.delay_factor = delay_factor
        self.delay_min = delay_min
        
        # # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            0,
            autospace(np.arange(n_item))
        )
        agent_policy = AssistantPolicy(action_state=action_state)

        # # Use default observation and inference engines
        observation_engine = None
        inference_engine = AssistantInferenceEngine()

        super().__init__(
            "assistant",
            *args,
            agent_policy=agent_policy,
            # policy_kwargs={"action_state": action_state},
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=agent_state,
            **kwargs
        )

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4

        :meta public:
        """

        self.state["box"][:] = np.zeros(self.n_item)
        self.state["due"][:] = np.zeros(self.n_item)

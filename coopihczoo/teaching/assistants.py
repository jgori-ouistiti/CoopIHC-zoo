from coopihc import BaseAgent, State, StateElement, Space, ExamplePolicy, \
    BasePolicy, autospace, BaseInferenceEngine
import numpy as np


# class Leitner:
#
#     def __init__(self, n_item, delay_factor, delay_min):
#
#         box = np.full(n_item, -1)
#         due = np.full(n_item, -1)
#
#         self.n_item = n_item
#
#         self.delay_factor = delay_factor
#         self.delay_min = delay_min
#
#         self.box = box
#         self.due = due
#
#     def update_box_and_due_time(self, last_idx,
#                                 last_was_success, last_time_reply):
#
#         if last_was_success:
#             self.box[last_idx] += 1
#         else:
#             self.box[last_idx] = \
#                 max(0, self.box[last_idx] - 1)
#
#         delay = self.delay_factor ** self.box[last_idx]
#         # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
#         self.due[last_idx] = \
#             last_time_reply + self.delay_min * delay

    # def _pickup_item(self, now):
    #
    #     seen = np.argwhere(np.asarray(self.box) >= 0).flatten()
    #     n_seen = len(seen)
    #
    #     if n_seen == self.n_item:
    #         return np.argmin(self.due)
    #
    #     else:
    #         seen__due = np.asarray(self.due)[seen]
    #         seen__is_due = np.asarray(seen__due) <= now
    #         if np.sum(seen__is_due):
    #             seen_and_is_due__due = seen__due[seen__is_due]
    #
    #             return seen[seen__is_due][np.argmin(seen_and_is_due__due)]
    #         else:
    #             return self._pickup_new()
    #
    # def _pickup_new(self):
    #     return np.argmin(self.box)
    #
    # def ask(self, now, last_was_success, last_time_reply, idx_last_q):
    #
    #     if idx_last_q is None:
    #         item_idx = self._pickup_new()
    #
    #     else:
    #
    #         self.update_box_and_due_time(
    #             last_idx=idx_last_q,
    #             last_was_success=last_was_success,
    #             last_time_reply=last_time_reply)
    #         item_idx = self._pickup_item(now)
    #
    #     return item_idx
    

class AssistantInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):

        item = int(self.observation["task_state"]["item"])
        timestamp = self.observation["task_state"]["timestamp"]
        
        last_was_success = self.observation["user_action"]["action"][0]
        print(type(last_was_success))
        
        if last_was_success:
            self.state["box"][last_idx, 0] += 1
        else:
            self.state["box"][last_idx] = \
                max(0, self.box[last_idx] - 1)

        delay = self.delay_factor ** self.state["box"][last_idx]
        # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
        self.self.state["due"][last_idx] = \
            last_time_reply + self.delay_min * delay

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

        box = self.state["box"].view(np.ndarray)
        due = self.state["due"].view(np.ndarray)

        if self.observation["task_state"]["iteration"] == 0:

            _action_value = box.argmin() # pickup new
        else:

            seen = np.argwhere(box >= 0).flatten()
            n_seen = len(seen)

            if n_seen == self.n_item:
                _action_value = np.argmin(due)

            else:
                seen__due = np.asarray(due)[seen]
                seen__is_due = np.asarray(seen__due) <= now
                if np.sum(seen__is_due):
                    seen_and_is_due__due = seen__due[seen__is_due]

                    _action_value  = seen[seen__is_due][np.argmin(seen_and_is_due__due)]
                else:
                    _action_value = box.argmin()  # pickup new

        return new_action, reward

    def reset(self):

        _action_value = -1
        self.action_state["action"][:] = _action_value


class Assistant(BaseAgent):

    def __init__(self, n_item, delay_factor, delay_min,
                 *args, **kwargs):

        # # Define an internal state with a 'goal' substate
        state = State()
        container = np.atleast_2d(np.zeros(n_item))
        state["box"] = StateElement(
            0,
            np.zeros_like(container),
            autospace(
                np.zeros_like(container),
                np.full(container.shape, np.inf),
                dtype=np.float32,
            ),
        )

        state["due"] = StateElement(
            0,
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
        agent_state = None

        # # Use default observation and inference engines
        observation_engine = None
        inference_engine = AssistantInferenceEngine()

        super().__init__(
            "user",
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
        pass
        # self.state["goal"][:] = 4

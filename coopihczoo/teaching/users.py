from coopihc import (
    BaseAgent,
    State,
    BasePolicy,
    RuleObservationEngine,
    BaseInferenceEngine,
    array_element,
    cat_element,
)
from coopihc.observation.utils import base_user_engine_specification
import numpy as np


class UserInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):

        item = int(self.observation["task_state"]["item"])
        timestamp = self.observation["task_state"]["timestamp"]

        self.state["last_pres_before_obs"][0, 0] = self.state["last_pres"][item, 0]
        self.state["n_pres_before_obs"][0, 0] = self.state["n_pres"][item, 0]

        self.state["last_pres"][item, 0] = timestamp
        self.state["n_pres"][item, 0] += 1

        reward = 0

        return self.state, reward


class UserPolicy(BasePolicy):
    """ExamplePolicy

    A simple policy which assumes that the agent using it has a 'goal' state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal.

    """

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    def sample(self, observation=None):

        """sample

        Compares 'x' to goal and issues +-1 accordingly.

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """

        param = self.host.param

        is_item_specific = bool(
            self.observation["task_state"]["is_item_specific"][0, 0]
        )

        item = int(self.observation["task_state"]["item"])
        timestamp = float(self.observation["task_state"]["timestamp"])

        n_pres = self.observation["user_state"]["n_pres_before_obs"].view(np.ndarray)[
            0, 0
        ]  # old and unique!!
        last_pres = self.observation["user_state"]["last_pres_before_obs"].view(
            np.ndarray
        )[
            0, 0
        ]  # old and unique!!

        reward = 0
        _action_value = 0
        # p = 0

        if n_pres > 0:

            if is_item_specific:
                init_forget = param[item, 0]
                rep_effect = param[item, 1]
            else:
                init_forget, rep_effect = param

            fr = init_forget * (1 - rep_effect) ** (n_pres - 1)

            delta = timestamp - last_pres

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                p = np.exp(-fr * delta)

            rv = np.random.random()

            _action_value = int(p > rv)

            # print("p", p, "rv", rv, "action_value", _action_value)

        else:
            pass
            # print("p", "item not seen!")

        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, reward

    def reset(self, random=True):

        _action_value = 0  # -1
        print(self.action_state["action"].shape)
        self.action_state["action"] = _action_value


class User(BaseAgent):
    """ """

    def __init__(self, param, *args, **kwargs):

        self.param = np.asarray(param)

        super().__init__("user", *args, **kwargs)

    def finit(self):

        n_item = int(self.bundle.task.state["n_item"][0, 0])
        is_item_specific = bool(self.bundle.task.state["is_item_specific"][0, 0])
        param = self.param

        self.state["n_pres"] = array_element(shape=(n_item,), low=-1, high=np.inf)
        self.state["last_pres"] = array_element(shape=(n_item,), low=-1, high=np.inf)

        self.state["n_pres_before_obs"] = array_element(shape=(1, ), low=-1, high=np.inf)
        self.state["last_pres_before_obs"] = array_element(shape=(1, ), low=-1, high=np.inf)

        self.state["param"] = array_element(shape=param.shape, low=-np.inf, high=np.inf, init=param.reshape(-1, 1))

        action_state = State()
        action_state["action"] = cat_element(N=2)

        observation_engine = RuleObservationEngine(
            deterministic_specification=base_user_engine_specification)
        inference_engine = UserInferenceEngine()
        agent_policy = UserPolicy(
            action_state=action_state, is_item_specific=is_item_specific, param=param)

        self._attach_policy(agent_policy)
        self._attach_observation_engine(observation_engine)
        self._attach_inference_engine(inference_engine)

    def reset(self, dic=None):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["n_pres"][:] = np.zeros(n_item)
        self.state["last_pres"][:] = np.zeros(n_item)
        self.state["n_pres_before_obs"][:] = 0
        self.state["last_pres_before_obs"][:] = 0
#
#
# class RLUserInferenceEngine(UserInferenceEngine):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def infer(self, user_state=None):
#         item = int(self.observation["task_state"]["item"])
#         timestamp = self.observation["task_state"]["timestamp"]
#
#         self.state["last_pres_before_obs"][0, 0] = self.state["last_pres"][item, 0]
#         self.state["n_pres_before_obs"][0, 0] = self.state["n_pres"][item, 0]
#
#         self.state["last_pres"][item, 0] = timestamp
#         self.state["n_pres"][item, 0] += 1
#
#
#
#         reward = 0
#
#         return self.state, reward
#
#
# class RLUser(User):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def finit(self):
#
#         super().finit()
#         n_item = int(self.bundle.task.state["n_item"][0, 0])
#
#         self.state["progress"] = array_element(shape=1, low=0, high=np.inf, init=0.0)
#         self.state["memory"] = array_element(shape=(n_item, 2), low=0, high=np.inf)
#
#         inf_engine = RLUserInferenceEngine
#         self.attach_inference_engine(inf_engine)
#




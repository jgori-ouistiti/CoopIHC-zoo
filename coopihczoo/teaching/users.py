from coopihc import (
    BaseAgent,
    State,
    BasePolicy,
    BaseInferenceEngine,
    array_element,
    discrete_array_element,
    cat_element,
)
from coopihc.observation.utils import base_user_engine_specification
import numpy as np


class UserInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):

        item = agent_observation["task_state"]["item"]
        timestamp = agent_observation["task_state"]["timestamp"]

        self.state["last_pres_before_obs"] = self.state["last_pres"][int(item)]
        self.state["n_pres_before_obs"] = self.state["n_pres"][int(item)]

        self.state["last_pres"][int(item)] = timestamp
        self.state["n_pres"][int(item)] += 1

        reward = 0
        print(self.state)
        return self.state, reward


class UserPolicy(BasePolicy):
    """ExamplePolicy

    A simple policy which assumes that the agent using it has a 'goal' state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal.

    """

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):

        """sample

        Compares 'x' to goal and issues +-1 accordingly.

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """

        param = self.host.param

        is_item_specific = self.parameters["is_item_specific"]

        item = agent_observation["task_state"]["item"]
        timestamp = agent_observation["task_state"]["timestamp"]

        n_pres = agent_observation["user_state"][
            "n_pres_before_obs"
        ]  # old and unique!!
        last_pres = agent_observation["user_state"][
            "last_pres_before_obs"
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

        else:
            pass

        return _action_value, reward

    def reset(self, random=True):

        _action_value = 0  # -1
        self.action_state["action"] = _action_value


class User(BaseAgent):
    """ """

    def __init__(self, param, *args, **kwargs):

        self.param = np.asarray(param)

        super().__init__("user", *args, **kwargs)

    def finit(self):

        n_item = self.parameters["n_item"]
        is_item_specific = self.parameters["is_item_specific"]

        self.state["n_pres"] = discrete_array_element(
            shape=(n_item,), low=-1, high=np.inf
        )
        self.state["last_pres"] = discrete_array_element(
            shape=(n_item,), low=-1, high=np.inf
        )

        self.state["n_pres_before_obs"] = discrete_array_element(low=-1, high=np.inf)
        self.state["last_pres_before_obs"] = discrete_array_element(low=-1, high=np.inf)

        param = self.param
        self.state["param"] = array_element(init=param, low=-np.inf, high=np.inf)

        action_state = State()
        action_state["action"] = cat_element(N=2)

        inference_engine = UserInferenceEngine()
        agent_policy = UserPolicy(
            action_state=action_state, is_item_specific=is_item_specific, param=param
        )

        self._attach_policy(agent_policy)
        self._attach_inference_engine(inference_engine)

    def reset(self, dic=None):

        n_item = self.parameters["n_item"]

        self.state["n_pres"] = np.zeros(n_item)
        self.state["last_pres"] = np.zeros(n_item)
        self.state["n_pres_before_obs"] = 0
        self.state["last_pres_before_obs"] = 0


#

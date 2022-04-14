from coopihc import (
    BaseAgent,
    State,
    BasePolicy,
    BaseInferenceEngine,
    array_element,
    discrete_array_element,
    cat_element,
)
from coopihczoo.teaching.memory_models import ExponentialDecayMemory
import numpy as np


# ============== Soft Refactor


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

            rv = self.get_rng().random()

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


##### ===================== Hard Refactor


class UserInferenceEngineWithP(BaseInferenceEngine):
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

        self.state["recall_probabilities"] = ExponentialDecayMemory.decay(
            delta_time=timestamp - self.state.last_pres,
            times_presented=self.state.n_pres - 1,
            initial_forgetting_rate=self.retention_params[:, 0],
            repetition_effect=self.retention_params[:, 1],
            log=False,
        )

        reward = 0
        return self.state, reward


class UserPolicyExternalModel(BasePolicy):
    """ExamplePolicy

    A simple policy which assumes that the agent using it has a 'goal' state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal.

    """

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):

        item = agent_observation["task_state"]["item"]
        n_pres = agent_observation["user_state"]["n_pres_before_obs"]

        reward = 0

        if n_pres > 0:  # If item seen before
            _action_value = int(
                self.state.recall_probabilities[int(item)] > self.get_rng().random()
            )

        else:  # If item never seen before
            _action_value = 0

        return _action_value, reward

    def reset(self, random=True):
        self.action_state["action"] = 0


class UserWithP(BaseAgent):
    """ """

    def __init__(self, param, *args, **kwargs):

        self.param = np.asarray(param)

        inference_engine = UserInferenceEngineWithP()
        observation_engine = None  # use default
        # Delay State and Policy to finit

        super().__init__(
            "user",
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            **kwargs
        )

    def finit(self):

        # set params --------- Hack for now
        self._parameters.update({"retention_params": self.param})

        # get params --------------
        n_item = self.parameters["n_item"]
        is_item_specific = self.parameters["is_item_specific"]

        # Set user state -------------
        self.state["n_pres"] = discrete_array_element(
            shape=(n_item,), low=-1, high=np.inf
        )
        self.state["last_pres"] = discrete_array_element(
            shape=(n_item,), low=-np.inf, high=np.inf
        )
        self.state["n_pres_before_obs"] = discrete_array_element(low=-1, high=np.inf)
        self.state["last_pres_before_obs"] = discrete_array_element(
            low=-np.inf, high=np.inf
        )
        self.state["recall_probabilities"] = array_element(
            low=0, high=1, shape=(n_item,), dtype=np.float64
        )

        # Set User action state ------------------
        action_state = State()
        action_state["action"] = cat_element(N=2)

        # Set User Policy
        agent_policy = UserPolicyExternalModel(
            action_state=action_state, is_item_specific=is_item_specific
        )

        self._attach_policy(agent_policy)

    def reset(self, dic=None):

        n_item = self.parameters["n_item"]

        self.state["n_pres"] = np.zeros(n_item)
        self.state["last_pres"] = np.full(
            (n_item,), -9223372036854774783
        )  # Hack for now
        self.state["n_pres_before_obs"] = 0
        self.state["last_pres_before_obs"] = 0
        self.state["recall_probabilities"] = np.zeros(n_item)

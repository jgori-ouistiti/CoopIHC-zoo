from coopihc import (
    BaseAgent,
    State,
    BasePolicy,
    BaseInferenceEngine,
    array_element,
    discrete_array_element,
    cat_element,
    BufferNotFilledError,
)
from coopihczoo.teaching.memory_models import ExponentialDecayMemory
import numpy as np


# ============== Soft Refactor


class ExponentialMemoryInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, buffer_depth=2, **kwargs)

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):
        item = agent_observation["task_state"]["item"]
        timestamp = agent_observation["task_state"]["timestamp"]

        try:
            self.state["last_pres_before_obs"] = self.buffer[-2]["task_state"][
                "last_pres"
            ][int(item)]
            self.state["n_pres_before_obs"] = self.buffer[-2]["task_state"]["n_pres"][
                int(item)
            ]
        except BufferNotFilledError:
            self.state["last_pres_before_obs"] = 0
            self.state["n_pres_before_obs"] = 0

        last_pres = agent_observation["task_state"]["last_pres"]
        n_pres = agent_observation["task_state"]["n_pres"]

        # cast to float to handle infinites
        self.state["recall_probabilities"] = ExponentialDecayMemory.decay(
            delta_time=(timestamp - last_pres.astype(np.float64)),
            times_presented=n_pres - 1,
            initial_forgetting_rate=self.retention_params[:, 0],
            repetition_effect=self.retention_params[:, 1],
            log=False,
        )

        reward = 0
        return self.state, reward


class UserPolicy(BasePolicy):
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


class ExponentialUser(BaseAgent):
    """ """

    def __init__(self, param, *args, **kwargs):

        self.param = np.asarray(param)

        inference_engine = ExponentialMemoryInferenceEngine()
        observation_engine = None  # use default
        # Delay State and Policy to finit

        super().__init__(
            "user",
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            **kwargs
        )

    def finit(self):

        # update params ---------
        self.update_parameters({"retention_params": self.param})

        # get params --------------
        n_item = self.parameters["n_item"]

        # Set user state -------------
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
        agent_policy = UserPolicy(action_state=action_state)

        self._attach_policy(agent_policy)

    def reset(self, dic=None):

        n_item = self.parameters["n_item"]

        self.state["n_pres_before_obs"] = 0
        self.state["last_pres_before_obs"] = 0
        self.state["recall_probabilities"] = np.zeros(n_item)

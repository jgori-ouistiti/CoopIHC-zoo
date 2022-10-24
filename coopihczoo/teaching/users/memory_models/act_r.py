import numpy as np
from scipy.special import expit
from coopihc import (
    BaseAgent,
    State,
    array_element,
    discrete_array_element,
    cat_element,
    BaseInferenceEngine,
)

EPS = np.finfo(np.float).eps


class ActRUser(BaseAgent):
    """ """

    def __init__(self, param, *args, n_iter=2048, **kwargs):

        self.param = np.asarray(param)

        inference_engine = ActRInferenceEngine()
        observation_engine = None  # use default
        self.n_iter = n_iter
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
        n_item = self.n_item

        # Set user state -------------

        self.state["ts"] = discrete_array_element(
            shape=(self.n_iter,), init=-1, dtype=np.int
        )  # timestamp
        self.state["hist"] = discrete_array_element(
            shape=(self.n_iter,), init=-1, dtype=np.int, low=-1, high=n_item
        )

        self.state["recall_probabilities"] = array_element(
            low=0, high=1, shape=(n_item,), dtype=np.float64
        )

        # Set User action state ------------------
        action_state = State()
        action_state["action"] = cat_element(N=2)

        # Set User Policy
        from coopihczoo.teaching.users.policy import UserPolicy

        agent_policy = UserPolicy(action_state=action_state)

        self._attach_policy(agent_policy)

    def reset(self, dic=None):

        n_iter = 2048  # Thank you, Julien

        n_item = self.parameters["n_item"]

        self.state.ts = np.full(shape=n_iter, fill_value=-1, dtype=np.int)
        self.state.hist = np.full(shape=n_iter, fill_value=-1, dtype=np.int)

        self.state.recall_probabilities = np.zeros(n_item)


class ActRInferenceEngine(BaseInferenceEngine):
    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):

        item = agent_observation["task_state"]["item"]
        timestamp = agent_observation["task_state"]["timestamp"]

        self.update(item=item, timestamp=timestamp)

        now = timestamp.astype(np.float64)

        # cast to float to handle infinities
        rp = self.recall_probabilities(
            now=now,
            log=False,
        )
        self.state["recall_probabilities"] = rp

        reward = 0
        return self.state, reward

    def p(self, item, param, now):

        tau, s, c, a = param

        hist = self.state.hist
        ts = self.state.ts

        b = hist == item
        rep = ts[b]
        n = len(rep)
        if n == 0:
            p = 0
        else:
            d = np.zeros(n)
            d[0] = a
            for i in range(1, n):
                delta_rep = rep[i] - rep[:i]
                e_m_rep = np.sum(np.power(delta_rep, -d[:i]))
                d[i] = c * e_m_rep + a  # Using previous em

            delta = now - rep
            with np.errstate(divide="ignore"):
                em = np.sum(np.power(delta, -d))

            x = (-tau + np.log(em)) / s
            p = expit(x)
        return p

    def update(self, item, timestamp):

        i = self.bundle.round_number

        self.state.hist[i - 1] = item
        self.state.ts[i - 1] = timestamp

    def recall_probabilities(
        self,
        now,
        log=False,
    ):

        param = self.host.param

        p = np.zeros(self.n_item)
        for i in range(self.n_item):
            p[i] = self.p(item=i, now=now, param=param)

        if log:
            return np.log(p + EPS)
        else:
            return p

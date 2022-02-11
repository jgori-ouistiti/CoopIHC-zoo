from coopihc import BaseAgent, State, StateElement, Space, BasePolicy, autospace, RuleObservationEngine, BaseInferenceEngine
from coopihc.observation.utils import base_user_engine_specification
import numpy as np


class UpdateMemory(BaseInferenceEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):

        item = int(self.observation["task_state"]["item"])
        timestamp = self.observation["task_state"]["timestamp"]

        print(self.state["last_pres"].shape)
        print(item)
        print("yo")
        print(type(self.state["last_pres"][0, 0]))

        self.state["last_pres"][0, 0] = timestamp
        print("done")
        self.state["n_pres"][item, 0] += 1

        reward = 0

        return self.state, reward


class ResponseGenerator(BasePolicy):
    """ExamplePolicy

    A simple policy which assumes that the agent using it has a 'goal' state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal.


    """

    def __init__(self, action_state, param, is_item_specific=False, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

        self.is_item_specific = is_item_specific
        self.param = param

    def sample(self, observation=None):
        """sample

        Compares 'x' to goal and issues +-1 accordingly.

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """

        item = self.observation["task_state"]["item"]
        timestamp = self.observation["task_state"]["timestamp"]

        param = self.param
        n_pres = self.observation["user_state"]["n_pres"]
        last_pres = self.observation["user_state"]["last_pres"]

        reward = 0
        _action_value = 0

        if n_pres[item] > 0:

            if self.is_item_specific:
                init_forget = param[item, 0]
                rep_effect = param[item, 1]
            else:
                init_forget, rep_effect = param

            fr = init_forget * (1 - rep_effect) ** (n_pres[item] - 1)
    #
            delta = timestamp - last_pres[item]

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                p = np.exp(- fr * delta)

            _action_value = p > np.random.random()

        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, reward


class User(BaseAgent):
    """
    """

    def __init__(self, n_item, is_item_specific, param,
                 *args, **kwargs):

        self.n_item = n_item

        # Define an internal state with a 'goal' substate
        state = State()

        #         self.n_pres = np.zeros(n_item, dtype=int)
        #         self.last_pres = np.zeros(n_item, dtype=float)

        n_pres_init = np.atleast_2d(np.zeros(n_item))
        state["n_pres"] = StateElement(
            np.zeros_like(n_pres_init),
            autospace(np.zeros_like(n_pres_init),
                      np.full(n_pres_init.shape, np.inf),
                      dtype=np.float32),
        )

        state["last_pres"] = StateElement(
            np.zeros_like(n_pres_init),
            autospace(np.zeros_like(n_pres_init),
                      np.full(n_pres_init.shape, np.inf),
                      dtype=np.float32),
        )

        # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            0,
            autospace([0, 1])
        )
        # agent_policy = ExamplePolicy(action_state=action_state)

        # Use default observation and inference engines
        observation_engine = RuleObservationEngine(
            deterministic_specification=base_user_engine_specification)
        inference_engine = UpdateMemory()
        policy = ResponseGenerator(action_state=action_state,
                                   is_item_specific=is_item_specific,
                                   param=param)

        super().__init__(
            "user",
            *args,
            agent_policy=policy,
            # policy_kwargs={"action_state": action_state},
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4

        :meta public:
        """
        self.state["n_pres"][:] = np.zeros(self.n_item)
        self.state["last_pres"][:] = np.zeros(self.n_item)

# class Exponential:
#
#     DUMMY_VALUE = -1
#
#     def __init__(self, n_item, n_iter):
#
#         self.n_item = n_item
#
#         self.seen = np.zeros(n_item, dtype=bool)
#         self.ts = np.full(n_iter, self.DUMMY_VALUE, dtype=float)
#         self.hist = np.full(n_iter, self.DUMMY_VALUE, dtype=int)
#         self.seen_item = None
#         self.n_seen = 0
#         self.i = 0
#
#         self.n_pres = np.zeros(n_item, dtype=int)
#         self.last_pres = np.zeros(n_item, dtype=float)
#
#     def p_seen(self, param, is_item_specific, now, cst_time):
#
#         seen = self.n_pres >= 1
#         if np.sum(seen) == 0:
#             return np.array([]), seen
#
#         if is_item_specific:
#             init_forget = param[seen, 0]
#             rep_effect = param[seen, 1]
#         else:
#             init_forget, rep_effect = param
#
#         fr = init_forget * (1 - rep_effect) ** (self.n_pres[seen] - 1)
#
#         last_pres = self.last_pres[seen]
#         delta = now - last_pres
#
#         delta *= cst_time
#         with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
#             p = np.exp(-fr * delta)
#         return p, seen
#
#     @staticmethod
#     def p_seen_spec_hist(param, now, hist, ts, seen, is_item_specific,
#                          cst_time):
#
#         if is_item_specific:
#             init_forget = param[seen, 0]
#             rep_effect = param[seen, 1]
#         else:
#             init_forget, rep_effect = param
#
#         seen_item = np.flatnonzero(seen)
#
#         n_seen = np.sum(seen)
#         n_pres = np.zeros(n_seen)
#         last_pres = np.zeros(n_seen)
#         for i, item in enumerate(seen_item):
#             is_item = hist == item
#             n_pres[i] = np.sum(is_item)
#             last_pres[i] = np.max(ts[is_item])
#
#         fr = init_forget * (1-rep_effect) ** (n_pres - 1)
#
#         delta = now - last_pres
#         delta *= cst_time
#         with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
#             p = np.exp(-fr * delta)
#         return p, seen
#
#     def log_lik_grid(self, item, grid_param, response, timestamp,
#                      cst_time):
#
#         fr = grid_param[:, 0] \
#              * (1 - grid_param[:, 1]) ** (self.n_pres[item] - 1)
#
#         delta = timestamp - self.last_pres[item]
#
#         delta *= cst_time
#         p_success = np.exp(- fr * delta)
#
#         p = p_success if response else 1-p_success
#
#         log_lik = np.log(p + EPS)
#         return log_lik
#
#     def p(self, item, param, now, is_item_specific, cst_time):
#
#         if is_item_specific:
#             init_forget = param[item, 0]
#             rep_effect = param[item, 1]
#         else:
#             init_forget, rep_effect = param
#
#         fr = init_forget * (1 - rep_effect) ** (self.n_pres[item] - 1)
#
#         delta = now - self.last_pres[item]
#
#         delta *= cst_time
#         with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
#             p = np.exp(- fr * delta)
#         return p
#
#     def update(self, item, timestamp):
#
#         self.last_pres[item] = timestamp
#         self.n_pres[item] += 1
#
#         self.hist[self.i] = item
#         self.ts[self.i] = timestamp
#
#         self.seen[item] = True
#
#         self.seen_item = np.flatnonzero(self.seen)
#         self.n_seen = np.sum(self.seen)
#
#         self.i += 1

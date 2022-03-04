from coopihc import BaseAgent, State, \
    cat_element, \
    array_element, \
    BasePolicy, BaseInferenceEngine, RuleObservationEngine, oracle_engine_specification
import numpy as np


class RlTeacherInferenceEngine(BaseInferenceEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):

        now = int(self.observation["task_state"]["timestamp"])
        log_thr = float(self.observation["task_state"]["log_thr"])

        is_item_specific = self.observation["user_state"]["param"][0, 0]

        n_pres = self.observation["user_state"]["n_pres"].view(np.ndarray).flatten()
        last_pres = self.observation["user_state"]["last_pres"].view(np.ndarray).flatten()

        seen = n_pres > 0
        unseen = np.invert(seen)
        delta = now - last_pres[seen]  # only consider already seen items
        rep = n_pres[seen] - 1.  # only consider already seen items

        if is_item_specific:
            init_forget_rate = self.observation["user_state"]["param"][:, 0]
            rep_effect = self.observation["user_state"]["param"][:, 1]

        else:
            init_forget_rate = self.observation["user_state"]["param"][0, 0]
            rep_effect = self.observation["user_state"]["param"][1, 0]

        if is_item_specific:
            forget_rate = \
                init_forget_rate[seen] * (1 - rep_effect) ** rep
        else:
            forget_rate = \
                init_forget_rate * (1 - rep_effect) ** rep

        survival = - (log_thr / forget_rate) - delta
        survival[survival < 0] = 0.

        if is_item_specific:
            init_forget_rate_seen = init_forget_rate[seen]
        else:
            init_forget_rate_seen = init_forget_rate

        seen_f_rate_if_action = init_forget_rate_seen * (1 - rep_effect) ** (rep + 1)
        seen_survival_if_action = - log_thr / seen_f_rate_if_action

        if is_item_specific:
            unseen_f_rate_if_action = init_forget_rate[unseen]
        else:
            unseen_f_rate_if_action = init_forget_rate

        unseen_survival_if_action = - log_thr / unseen_f_rate_if_action

        # self.memory_state[:, 0] = seen
        self.state["memory"][seen, 0] = survival
        self.state["memory"][unseen, 0] = 0.
        self.state["memory"][seen, 1] = seen_survival_if_action
        self.state["memory"][unseen, 1] = unseen_survival_if_action

        total_n = \
            self.observation["task_state"]["n_iter_per_ss"] \
            * self.observation["task_state"]["n_session"]

        current_iter = \
            int(self.observation["task_state"]["iteration"] \
            + self.observation["task_state"]["n_iter_per_ss"] * self.observation["task_state"]["session"])

        # print("total_n", int(total_n))
        # print("current_iter", current_iter)

        self.state["progress"][:] = current_iter / (total_n-1)

        # print("iteration", int(self.observation["task_state"]["iteration"]))
        # print("progress", float(self.state["progress"]))

        # self.memory_state[:, :] /= max_iter
        reward = 0
        return self.state, reward


class RlTeacherPolicy(BasePolicy):

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    # def sample(self, observation=None):
    #
    #     _action_value = 0
    #
    #     reward = 0
    #
    #     new_action = self.new_action
    #     new_action[:] = _action_value
    #
    #     return new_action, reward

    def reset(self, random=True):

        _action_value = 0
        self.action_state["action"][:] = _action_value


class Teacher(BaseAgent):

    def __init__(self, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)

    def finit(self, *args, **kwargs):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["progress"] = array_element(shape=1, low=0, high=np.inf, init=0.0)
        self.state["memory"] = array_element(shape=(n_item, 2), low=0, high=np.inf)

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(min=0, max=n_item)

        agent_policy = RlTeacherPolicy(action_state=action_state)

        # Inference engine
        inference_engine = RlTeacherInferenceEngine()

        # Use default observation engine
        observation_engine = RuleObservationEngine(
            deterministic_specification=oracle_engine_specification)

        self.attach_policy(agent_policy)
        self.attach_observation_engine(observation_engine)
        self.attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset.

        :meta public:
        """

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["progress"][:] = 0
        self.state["memory"][:] = np.zeros((n_item, 2))

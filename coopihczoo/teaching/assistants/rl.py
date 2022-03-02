from coopihc import BaseAgent, State, \
    cat_element, \
    array_element, \
    BasePolicy, BaseInferenceEngine, RuleObservationEngine, oracle_engine_specification
import numpy as np


class RlTeacherInferenceEngine(BaseInferenceEngine):

    def __init__(self, thr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_thr = np.log(thr)

    def infer(self, user_state=None):

        # last_item = int(self.observation["task_state"]["item"])
        # last_time_reply = self.observation["task_state"]["timestamp"]
        #
        # last_was_success = self.observation["user_action"]["action"][0]

        current_iter = self.observation["task_state"]["iteration"]
        max_iter = self.observation["task_state"]["max_iter"]

        init_forget_rate = self.observation["user_state"]["param"][0, 0]
        rep_effect = self.observation["user_state"]["param"][1, 0]

        n_pres = self.observation["user_state"]["n_pres"].view(np.ndarray).flatten()
        last_pres = self.observation["user_state"]["last_pres"].view(np.ndarray).flatten()

        seen = n_pres > 0
        unseen = np.invert(seen)
        delta = last_pres[seen]  # only consider already seen items
        rep = n_pres[seen] - 1.  # only consider already seen items

        # forget_rate = self.init_forget_rate[seen] * \
        #     (1 - self.rep_effect[seen]) ** rep

        forget_rate = init_forget_rate * (1 - rep_effect) ** rep

        # if self.current_iter == (self.n_iter_per_session - 1):
        #     # It will be a break before the next iteration
        #     delta += self.break_length
        # else:
        #     delta += self.time_per_iter

        survival = - (self.log_thr / forget_rate) - delta
        survival[survival < 0] = 0.

        seen_f_rate_if_action = init_forget_rate * (1 - rep_effect) ** (rep + 1)

        seen_survival_if_action = - self.log_thr / seen_f_rate_if_action

        unseen_f_rate_if_action = init_forget_rate   # [unseen]
        unseen_survival_if_action = - self.log_thr / unseen_f_rate_if_action

        # self.memory_state[:, 0] = seen
        self.state["memory"][seen, 0] = survival
        self.state["memory"][unseen, 0] = 0.
        self.state["memory"][seen, 1] = seen_survival_if_action
        self.state["memory"][unseen, 1] = unseen_survival_if_action

        self.state["progress"] = (current_iter + 1) / max_iter  # +1? Sure?

        # self.memory_state[:, :] /= max_iter

        reward = 0

        if current_iter == max_iter - 1:
            reward = np.sum(- forget_rate > self.log_thr)

        return self.state, reward


class RlTeacherPolicy(BasePolicy):

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    def sample(self, observation=None):

        _action_value = 0

        reward = 0
        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, reward

    def reset(self, random=True):

        _action_value = -1
        self.action_state["action"][:] = _action_value


class Teacher(BaseAgent):

    def __init__(self, thr, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)
        self.thr = thr

    def finit(self, *args, **kwargs):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["progress"] = array_element(shape=1, low=0, high=np.inf, init=0.0)
        self.state["memory"] = array_element(shape=(n_item, 2), low=0, high=np.inf)
        self.state["current_total_iter"] = array_element(shape=1, low=0, high=np.inf)

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(min=0, max=n_item)

        agent_policy = RlTeacherPolicy(action_state=action_state)

        # Inference engine
        inference_engine = RlTeacherInferenceEngine(thr=self.thr)

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

        pass

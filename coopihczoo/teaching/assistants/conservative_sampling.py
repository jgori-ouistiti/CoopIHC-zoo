from coopihc import BaseAgent, State, \
    cat_element, \
    BasePolicy, BaseInferenceEngine, RuleObservationEngine, oracle_engine_specification
import numpy as np


class ConservativeSamplingInferenceEngine(BaseInferenceEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):
        return super().infer(user_state=user_state)


class ConservativeSamplingPolicy(BasePolicy):

    def __init__(self, action_state, *args, **kwargs):
        super().__init__(action_state=action_state, *args, **kwargs)

    def _threshold_select(
            self, n_pres, initial_forget_rates,
            initial_repetition_rates, n_item,
            delta):

        if np.max(n_pres) == 0:
            item = 0
        else:
            seen = n_pres > 0

            log_p_seen = self._cp_log_p_seen(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=initial_forget_rates,
                initial_repetition_rates=initial_repetition_rates)

            if np.sum(seen) == n_item \
                    or np.min(log_p_seen) <= self.observation.task_state.log_thr:

                item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
            else:
                item = np.argmin(seen)

        return item

    def _cp_log_p_seen(
            self,
            n_pres,
            delta,
            initial_forget_rates,
            initial_repetition_rates):

        view = n_pres > 0
        rep = n_pres[view] - 1.
        delta = delta[view]

        if self.observation.task_state.is_item_specific:

            init_forget = initial_forget_rates[np.nonzero(view)]
            rep_effect = initial_repetition_rates[np.nonzero(view)]

        else:
            init_forget = initial_forget_rates
            rep_effect = initial_repetition_rates

        forget_rate = init_forget * (1 - rep_effect) ** rep
        logp_recall = - forget_rate * delta
        return logp_recall

    def step(
            self,
            item,
            n_pres,
            delta,
            current_iter,
            current_ss,
    ):

        done = False

        # update progression within session, and between session
        # - which iteration the learner is at?
        # - which session the learner is at?
        current_iter += 1
        if current_iter >= int(self.observation.task_state.n_iter_per_ss):
            current_iter = 0
            current_ss += 1
            time_elapsed = int(self.observation.task_state.break_length)
        else:
            time_elapsed = int(self.observation.task_state.inter_trial)

        if current_ss >= int(self.observation.task_state.n_session):
            done = True

        # increase delta
        delta += time_elapsed
        # ...specific for item shown
        delta[item] = time_elapsed
        # increment number of presentation
        n_pres[item] += 1

        return n_pres, delta, current_iter, current_ss, done

    def sample(self, observation=None):

        is_item_specific = bool(self.observation.task_state.is_item_specific)

        if is_item_specific:
            init_forget_rate = self.observation["user_state"]["param"][:, 0]
            rep_effect = self.observation["user_state"]["param"][:, 1]

        else:
            init_forget_rate = self.observation["user_state"]["param"][0, 0]
            rep_effect = self.observation["user_state"]["param"][1, 0]

        current_iter = int(self.observation.task_state.iteration)
        current_ss = int(self.observation.task_state.session)

        n_item = int(self.observation.task_state.n_item)

        log_thr = float(self.observation.task_state.log_thr)

        delta_current = \
            int(self.observation.task_state.timestamp) \
            - self.observation.user_state.last_pres.view(np.ndarray)
        n_pres_current = self.observation.user_state.n_pres.view(np.ndarray)

        # Reduce the number of item to learn
        # until every item presented is learnable
        while True:

            n_pres = n_pres_current[:n_item]
            delta = delta_current[:n_item]

            first_item = self._threshold_select(
                n_pres=n_pres,  delta=delta,
                initial_forget_rates=init_forget_rate,
                initial_repetition_rates=rep_effect,
                n_item=n_item)

            n_item = first_item + 1

            n_pres = n_pres_current[:n_item].copy()
            delta = delta_current[:n_item].copy()

            n_pres, delta, current_iter, current_ss, done = \
                self.step(
                    item=first_item,
                    n_pres=n_pres,
                    delta=delta,
                    current_iter=current_iter,
                    current_ss=current_ss)

            # Do rollouts...
            while not done:

                item = self._threshold_select(
                    n_pres=n_pres, delta=delta,
                    initial_forget_rates=init_forget_rate,
                    initial_repetition_rates=rep_effect,
                    n_item=n_item)

                n_pres, delta, current_iter, current_ss, done = \
                    self.step(item, n_pres, delta, current_iter,
                              current_ss)

            log_p_seen = self._cp_log_p_seen(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=init_forget_rate,
                initial_repetition_rates=rep_effect)

            n_learnt = np.sum(log_p_seen > log_thr)
            if n_learnt == n_item:
                break

            n_item = first_item
            if n_item <= 1:
                return 0

        _action_value = first_item

        new_action = self.new_action
        new_action[:] = _action_value

        reward = 0
        return new_action, reward

    def reset(self, random=True):

        _action_value = 0
        self.action_state["action"][:] = _action_value


class ConservativeSampling(BaseAgent):

    def __init__(self, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)

    def finit(self, *args, **kwargs):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(min=0, max=n_item)

        agent_policy = ConservativeSamplingPolicy(action_state=action_state)

        # Inference engine
        inference_engine = ConservativeSamplingInferenceEngine()

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

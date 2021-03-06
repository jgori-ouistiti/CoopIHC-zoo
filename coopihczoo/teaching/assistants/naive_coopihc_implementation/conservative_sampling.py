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

    def __init__(self, action_state,
                 n_session, inter_trial,
                 n_item, n_iter_per_ss,
                 break_length,
                 log_thr,
                 is_item_specific,
                 *args, **kwargs):

        super().__init__(action_state=action_state, *args, **kwargs)
        self.n_item = n_item
        self.n_session = n_session
        self.inter_trial = inter_trial
        self.n_iter_per_ss = n_iter_per_ss
        self.break_length = break_length
        self.log_thr = log_thr
        self.is_item_specific = is_item_specific

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
                    or np.min(log_p_seen) <= self.log_thr:

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

        if self.is_item_specific:

            init_forget = initial_forget_rates[np.nonzero(view)]
            rep_effect = initial_repetition_rates[np.nonzero(view)]

        else:
            init_forget = initial_forget_rates
            rep_effect = initial_repetition_rates

        forget_rate = init_forget * (1 - rep_effect) ** rep
        logp_recall = - forget_rate * delta
        return logp_recall

    def _step(
            self,
            item,
            n_pres,
            delta,
            current_iter,
            current_ss):

        done = False

        # update progression within session, and between session
        # - which iteration the learner is at?
        # - which session the learner is at?
        current_iter += 1
        if current_iter >= self.n_iter_per_ss:
            current_iter = 0
            current_ss += 1
            time_elapsed = self.break_length
        else:
            time_elapsed = self.inter_trial

        if current_ss >= self.n_session:
            done = True

        # increase delta
        delta += time_elapsed
        # ...specific for item shown
        delta[item] = time_elapsed
        # increment number of presentation
        n_pres[item] += 1

        return n_pres, delta, current_iter, current_ss, done

    def _loop(self,
              iteration, session,
              n_pres, timestamp,
              last_pres,
              init_forget_rate, rep_effect):

        n_item = self.n_item
        current_n_pres = n_pres
        current_iter = iteration
        current_ss = session
        current_delta = timestamp - last_pres

        # Reduce the number of item to learn
        # until every item presented is learnable
        while True:

            n_pres = current_n_pres[:n_item]
            delta = current_delta[:n_item]

            first_item = self._threshold_select(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=init_forget_rate,
                initial_repetition_rates=rep_effect,
                n_item=n_item)

            n_item = first_item + 1

            n_pres = current_n_pres[:n_item].copy()
            delta = current_delta[:n_item].copy()

            n_pres, delta, current_iter, current_ss, done = \
                self._step(
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
                    self._step(
                        item=item, n_pres=n_pres, delta=delta,
                        current_iter=current_iter,
                        current_ss=current_ss)

            log_p_seen = self._cp_log_p_seen(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=init_forget_rate,
                initial_repetition_rates=rep_effect)

            n_learnt = np.sum(log_p_seen > self.log_thr)
            if n_learnt == n_item:
                break

            n_item = first_item
            if n_item <= 1:
                return 0

        return first_item

    def sample(self, observation=None):
        if observation is None:
            observation = self.observation

        if self.is_item_specific:
            init_forget_rate = observation.user_state.param[:, 0]
            rep_effect = observation.user_state.param[:, 1]

        else:
            init_forget_rate = observation.user_state.param[0, 0]
            rep_effect = observation.user_state.param[1, 0]

        iteration = int(observation.task_state.iteration)
        session = int(observation.task_state.session)
        timestamp = int(observation.task_state.timestamp)
        last_pres = observation.user_state.last_pres.view(np.ndarray)
        n_pres = observation.user_state.n_pres.view(np.ndarray)

        first_item = self._loop(
            timestamp=timestamp,
            last_pres=last_pres,
            iteration=iteration,
            session=session,
            n_pres=n_pres,
            init_forget_rate=init_forget_rate,
            rep_effect=rep_effect)

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

        n_item = int(self.bundle.task.state.n_item[0, 0])
        n_session = int(self.bundle.task.state.n_session[0, 0])
        inter_trial = int(self.bundle.task.state.inter_trial[0, 0])
        n_iter_per_ss = int(self.bundle.task.state.n_iter_per_ss[0, 0])
        break_length = int(self.bundle.task.state.break_length[0, 0])
        log_thr = float(self.bundle.task.state.log_thr[0, 0])
        is_item_specific = bool(self.bundle.task.state.is_item_specific[0, 0])

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(min=0, max=n_item)

        agent_policy = ConservativeSamplingPolicy(
            action_state=action_state,
            n_session=n_session, inter_trial=inter_trial,
            n_item=n_item, n_iter_per_ss=n_iter_per_ss,
            break_length=break_length,
            log_thr=log_thr,
            is_item_specific=is_item_specific)

        # Inference engine
        inference_engine = ConservativeSamplingInferenceEngine()

        # Use default observation engine
        observation_engine = RuleObservationEngine(
            deterministic_specification=oracle_engine_specification)

        self.attach_policy(agent_policy)
        self.attach_observation_engine(observation_engine)
        self.attach_inference_engine(inference_engine)

    def reset(self, dic=None):
        pass

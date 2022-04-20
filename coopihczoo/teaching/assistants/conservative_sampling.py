from coopihc import (
    BaseAgent,
    State,
    cat_element,
    BasePolicy,
    BaseInferenceEngine,
    DualInferenceEngine,
    Simulator,
    array_element,
    BufferNotFilledError,
    DualPolicy,
)
import numpy as np

from coopihczoo.teaching.assistants.myopic import MyopicPolicy
import copy


class ConservativeSampling(BaseAgent):
    def __init__(
        self, task_class, user_class, task_kwargs={}, user_kwargs={}, **kwargs
    ):
        class NewInferenceEngine(BaseInferenceEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            @BaseInferenceEngine.default_value
            def infer(self, agent_observation=None):
                return super().infer(agent_observation=agent_observation)

        inference_engine = DualInferenceEngine(
            primary_inference_engine=InferUserPRecall(),
            dual_inference_engine=NewInferenceEngine(),
            primary_kwargs={},
            dual_kwargs={},
            buffer_depth=2,
        )

        super().__init__("assistant", agent_inference_engine=inference_engine, **kwargs)

        self.task_class = task_class
        self.user_class = user_class
        self.task_kwargs = task_kwargs
        self.user_kwargs = user_kwargs

        self.task_model = task_class(**task_kwargs)
        self.user_model = user_class(**user_kwargs)

        self.simulator = Simulator(
            task_model=self.task_model,
            user_model=self.user_model,
            assistant=self,
        )

    def finit(self, *args, **kwargs):

        n_item = self.parameters["n_item"]
        self.state["user_estimated_recall_probabilities"] = array_element(
            init=np.zeros((n_item,)), low=0, high=1, dtype=np.float64
        )

        # ================= Policy ============

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)

        agent_policy = BasePolicy(action_state=action_state)

        # agent_policy = DualPolicy(
        #     primary_policy=ConservativeSamplingPolicy(
        #         self.task_class, self.user_class, action_state
        #     ),
        #     dual_policy=MyopicPolicy(copy.deepcopy(action_state)),
        # )

        # ================= Inference Engine =========

        self._attach_policy(agent_policy)

    def reset(self, dic=None):
        pass


class InferUserPRecall(BaseInferenceEngine):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(buffer_depth=2, **kwargs)

        self._inference_count = 0

    @property
    def simulator(self):
        return self.host.simulator

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):

        # First time, nothing do no
        self._inference_count += 1
        if self._inference_count == 1:
            return self.state, 0

        agent_state = getattr(agent_observation, f"{self.role}_state")

        # ========== Set the simulator in the state just before user inference by the user model
        reset_dic = copy.deepcopy(agent_observation)
        reset_dic["game_info"][
            "turn_index"
        ] = 0  # Set turn to just before user observation and inference
        reset_dic["user_state"] = {}
        reset_dic["user_state"]["recall_probabilities"] = copy.deepcopy(
            agent_state["user_estimated_recall_probabilities"]
        )  # Plug in the assistant's estimated probabilities

        # ----------- fill in user n pres and last pres based on second last observation (i.e. on the assistant observation that was just before the user's observation)
        try:
            reset_dic["user_state"]["n_pres"] = self.buffer[-2]["task_state"]["n_pres"]
            reset_dic["user_state"]["last_pres"] = self.buffer[-2]["task_state"][
                "last_pres"
            ]
        except BufferNotFilledError:  # Deal with start edge case
            reset_dic["user_state"]["n_pres"] = np.zeros((self.n_item,))
            reset_dic["user_state"]["last_pres"] = 0

        # Open simulator (switch do duals)
        self.simulator.open()
        self.simulator.reset(dic=reset_dic)
        self.simulator.quarter_step()  # just perform observation and inference by user model
        recall_probs = self.simulator.state.user_state["recall_probabilities"]
        self.simulator.close()
        # close simulator (switch to primaries)

        self.state[
            "user_estimated_recall_probabilities"
        ] = recall_probs  # update assistant internal state
        return self.state, 0


# class ConservativeSamplingPolicy(BasePolicy):
#     def __init__(
#         self,
#         task_class,
#         user_model,
#         action_state,
#         task_class_kwargs={},
#         user_class_kwargs={},
#         **kwargs
#     ):
#         super().__init__(action_state=action_state, **kwargs)
#         self.task_class = task_class
#         self.user_model = user_model
#         self.task_class_kwargs = task_class_kwargs
#         self.user_class_kwargs = user_class_kwargs

#     @BasePolicy.default_value
#     def sample(self, agent_observation=None, agent_state=None):
#         current_trial = agent_observation.game_info.round_index
#         current_game_state =
#         new_task_class_kwargs = self.task_class_kwargs

#         while True:
#             task = self.task_class(**new_task_class_kwargs)
#             user = self.user_class(**self.user_class_kwargs)


# class ConservativeSamplingPolicy(BasePolicy):
#     def __init__(
#         self,
#         action_state,
#         n_session,
#         inter_trial,
#         n_item,
#         n_iter_per_ss,
#         break_length,
#         log_thr,
#         is_item_specific,
#         *args,
#         **kwargs
#     ):

#         super().__init__(action_state=action_state, *args, **kwargs)
#         self.n_item = n_item
#         self.n_session = n_session
#         self.inter_trial = inter_trial
#         self.n_iter_per_ss = n_iter_per_ss
#         self.break_length = break_length
#         self.log_thr = log_thr
#         self.is_item_specific = is_item_specific

#     def _threshold_select(
#         self, n_pres, initial_forget_rates, initial_repetition_rates, n_item, delta
#     ):

#         if np.max(n_pres) == 0:
#             item = 0
#         else:
#             seen = n_pres > 0

#             log_p_seen = self._cp_log_p_seen(
#                 n_pres=n_pres,
#                 delta=delta,
#                 initial_forget_rates=initial_forget_rates,
#                 initial_repetition_rates=initial_repetition_rates,
#             )

#             if np.sum(seen) == n_item or np.min(log_p_seen) <= self.log_thr:

#                 item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
#             else:
#                 item = np.argmin(seen)

#         return item

#     def _cp_log_p_seen(
#         self, n_pres, delta, initial_forget_rates, initial_repetition_rates
#     ):

#         view = n_pres > 0
#         rep = n_pres[view] - 1.0
#         delta = delta[view]

#         if self.is_item_specific:

#             init_forget = initial_forget_rates[np.nonzero(view)]
#             rep_effect = initial_repetition_rates[np.nonzero(view)]

#         else:
#             init_forget = initial_forget_rates
#             rep_effect = initial_repetition_rates

#         forget_rate = init_forget * (1 - rep_effect) ** rep
#         logp_recall = -forget_rate * delta
#         return logp_recall

#     def _step(self, item, n_pres, delta, current_iter, current_ss):

#         done = False

#         # update progression within session, and between session
#         # - which iteration the learner is at?
#         # - which session the learner is at?
#         current_iter += 1
#         if current_iter >= self.n_iter_per_ss:
#             current_iter = 0
#             current_ss += 1
#             time_elapsed = self.break_length
#         else:
#             time_elapsed = self.inter_trial

#         if current_ss >= self.n_session:
#             done = True

#         # increase delta
#         delta += time_elapsed
#         # ...specific for item shown
#         delta[item] = time_elapsed
#         # increment number of presentation
#         n_pres[item] += 1

#         return n_pres, delta, current_iter, current_ss, done

#     def _loop(
#         self,
#         iteration,
#         session,
#         n_pres,
#         timestamp,
#         last_pres,
#         init_forget_rate,
#         rep_effect,
#     ):

#         n_item = self.n_item
#         current_n_pres = n_pres
#         current_iter = iteration
#         current_ss = session
#         current_delta = timestamp - last_pres

#         # Reduce the number of item to learn
#         # until every item presented is learnable
#         while True:

#             n_pres = current_n_pres[:n_item]
#             delta = current_delta[:n_item]

#             first_item = self._threshold_select(
#                 n_pres=n_pres,
#                 delta=delta,
#                 initial_forget_rates=init_forget_rate,
#                 initial_repetition_rates=rep_effect,
#                 n_item=n_item,
#             )

#             n_item = first_item + 1

#             n_pres = current_n_pres[:n_item].copy()
#             delta = current_delta[:n_item].copy()

#             n_pres, delta, current_iter, current_ss, done = self._step(
#                 item=first_item,
#                 n_pres=n_pres,
#                 delta=delta,
#                 current_iter=current_iter,
#                 current_ss=current_ss,
#             )

#             # Do rollouts...
#             while not done:
#                 item = self._threshold_select(
#                     n_pres=n_pres,
#                     delta=delta,
#                     initial_forget_rates=init_forget_rate,
#                     initial_repetition_rates=rep_effect,
#                     n_item=n_item,
#                 )

#                 n_pres, delta, current_iter, current_ss, done = self._step(
#                     item=item,
#                     n_pres=n_pres,
#                     delta=delta,
#                     current_iter=current_iter,
#                     current_ss=current_ss,
#                 )

#             log_p_seen = self._cp_log_p_seen(
#                 n_pres=n_pres,
#                 delta=delta,
#                 initial_forget_rates=init_forget_rate,
#                 initial_repetition_rates=rep_effect,
#             )

#             n_learnt = np.sum(log_p_seen > self.log_thr)
#             if n_learnt == n_item:
#                 break

#             n_item = first_item
#             if n_item <= 1:
#                 return 0

#         return first_item

#     def sample(self, observation=None, **kwargs):
#         if observation is None:
#             observation = self.observation

#         if self.is_item_specific:
#             init_forget_rate = observation.user_state.param[:, 0]
#             rep_effect = observation.user_state.param[:, 1]

#         else:
#             init_forget_rate = observation.user_state.param[0]
#             rep_effect = observation.user_state.param[1]

#         iteration = int(observation.task_state.iteration)
#         session = int(observation.task_state.session)
#         timestamp = int(observation.task_state.timestamp)
#         last_pres = observation.user_state.last_pres.view(np.ndarray)
#         n_pres = observation.user_state.n_pres.view(np.ndarray)

#         _action_value = self._loop(
#             timestamp=timestamp,
#             last_pres=last_pres,
#             iteration=iteration,
#             session=session,
#             n_pres=n_pres,
#             init_forget_rate=init_forget_rate,
#             rep_effect=rep_effect,
#         )

#         reward = 0
#         return _action_value, reward

#     def reset(self, random=True):

#         self.action_state["action"] = 0


# class ConservativeSampling(BaseAgent):
#     def __init__(self, *args, **kwargs):
#         super().__init__("assistant", *args, **kwargs)

#     def finit(self, *args, **kwargs):

#         n_item = self.parameters.n_item
#         n_session = int(self.bundle.task.state.n_session)
#         inter_trial = int(self.bundle.task.state.inter_trial)
#         n_iter_per_ss = int(self.bundle.task.state.n_iter_per_ss)
#         break_length = int(self.bundle.task.state.break_length)
#         log_thr = float(self.bundle.task.state.log_thr)
#         is_item_specific = bool(self.bundle.task.state.is_item_specific)

#         # Call the policy defined above
#         action_state = State()
#         action_state["action"] = cat_element(N=n_item)

#         agent_policy = ConservativeSamplingPolicy(
#             action_state=action_state,
#             n_session=n_session,
#             inter_trial=inter_trial,
#             n_item=n_item,
#             n_iter_per_ss=n_iter_per_ss,
#             break_length=break_length,
#             log_thr=log_thr,
#             is_item_specific=is_item_specific,
#         )

#         # Use default observation engine
#         observation_engine = RuleObservationEngine(
#             deterministic_specification=oracle_engine_specification
#         )

#         self._attach_policy(agent_policy)
#         self._attach_observation_engine(observation_engine)
#         # self.attach_inference_engine(inference_engine)

#     def reset(self, dic=None):
#         pass

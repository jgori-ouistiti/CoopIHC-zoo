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
from coopihczoo.teaching.assistants.userPestimator import UserPEstimator
from coopihczoo.teaching.envs import TeachingOrchestrator

import copy


class ConservativeSampling(UserPEstimator):
    def __init__(
        self,
        task_class,
        user_class,
        teaching_orchestrator_kwargs,
        task_kwargs={},
        user_kwargs={},
        **kwargs,
    ):
        super().__init__(
            task_class,
            user_class,
            task_kwargs=task_kwargs,
            user_kwargs=user_kwargs,
            **kwargs,
        )
        self.parameters = teaching_orchestrator_kwargs

    def finit(self, *args, **kwargs):

        n_item = self.parameters["n_item"]
        self.state["user_estimated_recall_probabilities"] = array_element(
            init=np.zeros((n_item,)), low=0, high=1, dtype=np.float64
        )

        # ================= Policy ============

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=n_item)

        # agent_policy = BasePolicy(action_state=action_state)

        agent_policy = DualPolicy(
            # primary_policy=MyopicPolicy(action_state=copy.deepcopy(action_state)),
            primary_policy=ConservativeSamplingPolicy(
                self.task_class,
                self.user_class,
                action_state,
                task_class_kwargs=self.task_kwargs,
                user_class_kwargs=self.user_kwargs,
            ),
            dual_policy=MyopicPolicy(action_state=copy.deepcopy(action_state)),
        )

        self._attach_policy(agent_policy)

    def reset(self, dic=None):
        pass


class ConservativeSamplingPolicy(BasePolicy):
    def __init__(
        self,
        task_class,
        user_class,
        action_state,
        task_class_kwargs={},
        user_class_kwargs={},
        **kwargs,
    ):
        super().__init__(action_state=action_state, **kwargs)
        self.task_class = task_class
        self.user_class = user_class
        self.task_class_kwargs = task_class_kwargs
        self.user_class_kwargs = user_class_kwargs

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):
        current_iteration = int(agent_observation.game_info.round_index)
        if current_iteration == 0:  # First item
            return 0, 0

        # ============   Creating orchestrator schedule
        # use the original schedule, and modify it by removing the current iteration from it (respecting breaks as well)
        iterations_in_schedule = np.cumsum(self.n_iter_per_ss).tolist()
        _appended_iterations_in_schedule = iterations_in_schedule + [current_iteration]
        index = sorted(_appended_iterations_in_schedule).index(current_iteration)
        new_n_iter_per_ss = [
            iterations_in_schedule[index] - current_iteration
        ] + self.n_iter_per_ss[index + 1 :]
        new_breaks = self.breaks[index:]

        orchestrator_kwargs = {
            "n_iter_per_ss": new_n_iter_per_ss,
            "breaks": new_breaks,
            "time_before_exam": self.time_before_exam,
            "exam_threshold": self.exam_threshold,
        }
        # ============ Create game_state to which the game will be reset the first time
        game_reset_state = copy.deepcopy(
            agent_observation
        )  # Deepcopy just to be sure there is no interaction

        # Since user state is not observable, recreate it here from the assistant's knowledge ================
        game_reset_state["user_state"] = State()
        user_probs = game_reset_state.pop("assistant_state").pop(
            "user_estimated_recall_probabilities"
        )
        game_reset_state["user_state"]["recall_probabilities"] = user_probs
        try:
            last_item = int(agent_observation["task_state"]["item"])
            past_observation = self.host.inference_engine.buffer[-2]
            user_last_pres_before_obs = past_observation["task_state"]["last_pres"][
                last_item
            ]
            user_n_pres_before_obs = past_observation["task_state"]["n_pres"][last_item]
        except BufferNotFilledError:  # Deal with start edge case
            user_n_pres_before_obs = 0
            user_last_pres_before_obs = 0

        game_reset_state["user_state"]["n_pres_before_obs"] = user_n_pres_before_obs
        game_reset_state["user_state"][
            "last_pres_before_obs"
        ] = user_last_pres_before_obs
        # ============================= End recreating user state

        # =============== Init for conservative sampling
        new_task_class_kwargs = copy.deepcopy(self.task_class_kwargs)
        new_user_class_kwargs = copy.deepcopy(self.user_class_kwargs)
        n_item = self.n_item
        while True:
            simulator = Simulator(
                task_model=self.task_class(**new_task_class_kwargs),
                user_model=self.user_class(**new_user_class_kwargs),
                assistant=self.host,
                use_primary_inference=False,
                seed=1234,
                random_reset=False,
            )
            simulator.open()
            orchestrator = TeachingOrchestrator(simulator, **orchestrator_kwargs)
            orchestrator.reset(dic=copy.deepcopy(game_reset_state))
            if (
                orchestrator.raw_bundle.assistant.policy.mode != "dual"
                and orchestrator.raw_bundle.assistant.policy.dual_policy.__class__.__name__
                == "MyopicPolicy"
            ):
                raise RuntimeError(
                    f"The orchestrator is not using the correct policy. Should be in dual mode with MyopicPolicy, but it is in {orchestrator.raw_bundle.assistant.policy.mode} mode instead"
                )
            k = 0
            while True:
                k += 1
                state, rewards, is_done = orchestrator.step()
                if k == 1:  # Remember which item was chosen as first item
                    item_selected = state["assistant_action"]["action"]
                if is_done:
                    break
            if int(np.sum(list(rewards.values()))) == n_item:
                break
            else:
                n_pres_tmp = copy.deepcopy(game_reset_state["task_state"]["n_pres"])

                indices_keep = np.arange(n_item) != item_selected

                del game_reset_state["task_state"]["n_pres"]
                game_reset_state["task_state"]["n_pres"] = discrete_array_element(
                    init=n_pres_tmp[indices_keep]
                )
                n_item += -1
                new_task_class_kwargs["n_item"] = n_item
                new_user_class_kwargs["param"] = new_user_class_kwargs["param"][
                    indices_keep, :
                ]
        # while True:
        # Create simulator
        # -
        # Create orchestrator

        # run
        # if all items learned: break

        # new_task_class_kwargs = self.task_class_kwargs

        #     task = self.task_class(**new_task_class_kwargs)
        #     user = self.user_class(**self.user_class_kwargs)


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

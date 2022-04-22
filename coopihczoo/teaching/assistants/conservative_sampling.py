from cgitb import reset
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
    discrete_array_element,
)
import numpy as np

from coopihczoo.teaching.assistants.myopic import MyopicPolicy, Myopic
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

        agent_policy = ConservativeSamplingPolicy(
            self.task_class,
            self.user_class,
            action_state,
            task_class_kwargs=self.task_kwargs,
            user_class_kwargs=self.user_kwargs,
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
        self.presented_items = set()

    def _reduce(
        self,
        indices,
        game_reset_state_unreduced,
        task_args,
        user_args,
    ):
        reset_dic = self._reduce_reset_dic(indices, game_reset_state_unreduced)
        task_args, user_args = self._reduce_models(
            indices, task_args=task_args, user_args=user_args
        )
        n_item = len(indices)

        return reset_dic, n_item, task_args, user_args

    def _reduce_reset_dic(self, indices_keep, reset_dic):
        n_pres_tmp = np.asarray(reset_dic["task_state"]["n_pres"])
        last_pres_tmp = np.asarray(reset_dic["task_state"]["last_pres"])
        recall_probs = np.asarray(reset_dic["user_state"]["recall_probabilities"])

        del reset_dic["task_state"]["n_pres"]
        del reset_dic["task_state"]["last_pres"]
        del reset_dic["user_state"]["recall_probabilities"]

        reset_dic["task_state"]["n_pres"] = discrete_array_element(
            init=n_pres_tmp[indices_keep],
            low=-1,
        )
        reset_dic["task_state"]["last_pres"] = discrete_array_element(
            init=last_pres_tmp[indices_keep],
        )
        reset_dic["user_state"]["recall_probabilities"] = array_element(
            init=recall_probs[indices_keep], low=0, high=1, dtype=np.float64
        )
        return reset_dic

    def _reduce_models(self, indices_keep, task_args={}, user_args={}):
        # reduce the input arguments for the task and user classes
        task_args["n_item"] = len(indices_keep)
        user_args["param"] = user_args["param"][indices_keep, :]
        return task_args, user_args

    # def _reduce(self, item_selected, n_item, reset_dic, task_args, user_args):
    #     indices_keep = np.arange(n_item) != item_selected

    #     reset_dic = self._reduce_reset_dic(indices_keep, reset_dic)
    #     n_item, task_args, user_args = self._reduce_models(
    #         indices_keep, n_item, task_args=task_args, user_args=user_args
    #     )
    #     return n_item, reset_dic, task_args, user_args

    # def _reduce_models(self, indices_keep, n_item, task_args={}, user_args={}):
    #     # reduce the input arguments for the task and user classes
    #     n_item += -1
    #     task_args["n_item"] = n_item
    #     user_args["param"] = user_args["param"][indices_keep, :]
    #     return n_item, task_args, user_args

    # def _reduce_reset_dic(self, indices_keep, reset_dic):
    #     n_pres_tmp = np.asarray(reset_dic["task_state"]["n_pres"])
    #     last_pres_tmp = np.asarray(reset_dic["task_state"]["last_pres"])
    #     recall_probs = np.asarray(reset_dic["user_state"]["recall_probabilities"])

    #     del reset_dic["task_state"]["n_pres"]
    #     del reset_dic["task_state"]["last_pres"]
    #     del reset_dic["user_state"]["recall_probabilities"]

    #     reset_dic["task_state"]["n_pres"] = discrete_array_element(
    #         init=n_pres_tmp[indices_keep],
    #         low=-1,
    #     )
    #     reset_dic["task_state"]["last_pres"] = discrete_array_element(
    #         init=last_pres_tmp[indices_keep],
    #     )
    #     reset_dic["user_state"]["recall_probabilities"] = array_element(
    #         init=recall_probs[indices_keep], low=0, high=1, dtype=np.float64
    #     )
    #     return reset_dic

    def construct_reset_state_for_simu(self, agent_observation_copy):
        # Remove user and assistant actions, since not needed and to be sure there is no interaction
        del agent_observation_copy["user_action"]
        del agent_observation_copy["assistant_action"]
        # load estimated probs into user state
        agent_observation_copy["user_state"] = State()
        user_probs = agent_observation_copy.pop("assistant_state").pop(
            "user_estimated_recall_probabilities"
        )
        # agent_observation_copy["user_state"]["recall_probabilities"] = user_probs
        ### ====== BIG HACK ======
        agent_observation_copy["user_state"][
            "recall_probabilities"
        ] = self.host.bundle.user.state.recall_probabilities

        # load n_pres and last_pres into user state
        try:
            last_item = int(agent_observation_copy["task_state"]["item"])
            past_observation = self.host.inference_engine.buffer[-2]
            user_last_pres_before_obs = past_observation["task_state"]["last_pres"][
                last_item
            ]
            user_n_pres_before_obs = past_observation["task_state"]["n_pres"][last_item]
        except BufferNotFilledError:  # Deal with start edge case
            user_n_pres_before_obs = 0
            user_last_pres_before_obs = 0

        agent_observation_copy["user_state"][
            "n_pres_before_obs"
        ] = user_n_pres_before_obs
        agent_observation_copy["user_state"][
            "last_pres_before_obs"
        ] = user_last_pres_before_obs

        return agent_observation_copy

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):
        current_iteration = int(agent_observation.game_info.round_index)
        if current_iteration == 0:  # First item
            self.presented_items.add(0)
            return 0, 0

        # ============   new (reduced) schedule
        new_n_iter_per_ss, new_breaks = TeachingOrchestrator.reduce_schedule(
            self.n_iter_per_ss, self.breaks, current_iteration
        )
        print(current_iteration, new_n_iter_per_ss, new_breaks)
        orchestrator_kwargs = {
            "n_iter_per_ss": new_n_iter_per_ss,
            "breaks": new_breaks,
            "time_before_exam": self.time_before_exam,
            "exam_threshold": self.exam_threshold,
            "inter_trial": self.inter_trial,
        }
        # orchestrator_kwargs = {
        #     "n_iter_per_ss": self.n_iter_per_ss,
        #     "breaks": self.breaks,
        #     "time_before_exam": self.time_before_exam,
        #     "exam_threshold": self.exam_threshold,
        #     "inter_trial": self.inter_trial,
        # }
        # ============ Create game_state to which the simulation will be reset to
        game_reset_state = self.construct_reset_state_for_simu(
            copy.deepcopy(agent_observation)
        )  # Deepcopy just to be sure there is no interaction

        # start simulation to check if myopic policy will lead to all items preented being learned
        n_item = self.n_item
        simulator = Simulator(
            task_model=self.task_class(**self.task_class_kwargs),
            user_model=self.user_class(**self.user_class_kwargs),
            assistant=Myopic(
                override_agent_policy=MyopicPolicy(
                    action_state=State(**{"action": cat_element(n_item)})
                )
            ),
            use_primary_inference=False,
            seed=1234,
            random_reset=False,
        )
        simulator.open()
        orchestrator = TeachingOrchestrator(simulator, **orchestrator_kwargs)
        orchestrator.reset(dic=copy.deepcopy(game_reset_state))
        print(
            f"simulator round: {int(orchestrator.raw_bundle.state.game_info.round_index)}"
        )
        k = 0
        while True:
            # print(k + current_iteration)
            k += 1
            state, rewards, is_done = orchestrator.step()
            if k == 1:  # Remember which item was chosen as first item
                item_selected = int(state["assistant_action"]["action"])
            if is_done:
                break
        simulator.close()

        if int(np.sum(list(rewards.values()))) >= len(
            self.presented_items & set([item_selected])
        ):  # if all presented items (new one included) are remembered
            self.presented_items.add(item_selected)
            print(f"student can learn items presented{self.presented_items}")

        else:  # select item using myopic from the presented_items
            print("will not be able to learn")
            # make reduced model using only presented items
            reset_dic, n_item, task_args, user_args = self._reduce(
                list(self.presented_items),
                game_reset_state,
                copy.deepcopy(self.task_class_kwargs),
                copy.deepcopy(self.user_class_kwargs),
            )
            # create simulator, (no need for orchestrator since Myopic doesn't plan)
            simulator = Simulator(
                task_model=self.task_class(**task_args),
                user_model=self.user_class(**user_args),
                assistant=Myopic(
                    override_agent_policy=MyopicPolicy(
                        action_state=State(**{"action": cat_element(n_item)})
                    )
                ),
                use_primary_inference=False,
                seed=1234,
                random_reset=False,
            )
            simulator.open()

            simulator.reset(dic=copy.deepcopy(reset_dic))
            # single step
            state, rewards, is_done = simulator.step()
            simulator.close()
            item_selected = int(state["assistant_action"]["action"])

        return item_selected, 0


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

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

    def _reduce(self, item_selected, n_item, reset_dic, task_args, user_args):
        indices_keep = np.arange(n_item - 1)
        reset_dic = self._reduce_reset_dic(indices_keep, reset_dic)
        n_item, task_args, user_args = self._reduce_models(
            indices_keep, n_item, task_args=task_args, user_args=user_args
        )
        return n_item, reset_dic, task_args, user_args

    def _reduce_models(self, indices_keep, n_item, task_args={}, user_args={}):
        # reduce the input arguments for the task and user classes
        n_item = len(indices_keep)
        task_args["n_item"] = n_item
        user_args["param"] = user_args["param"][indices_keep, :]
        return n_item, task_args, user_args

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

    def simulate_myopic(
        self,
        n_item,
        reset_dic,
        task_class_kwargs,
        user_class_kwargs,
        orchestrator_kwargs,
    ):
        simulator = Simulator(
            task_model=self.task_class(**task_class_kwargs),
            user_model=self.user_class(**user_class_kwargs),
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
        orchestrator.reset(dic=copy.deepcopy(reset_dic))

        k = 0
        while True:
            k += 1
            state, rewards, is_done = orchestrator.step()
            if k == 1:  # Remember which item was chosen as first item
                item_selected = int(state["assistant_action"]["action"])
            if is_done:
                break
        simulator.close()

        return item_selected, rewards

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

        orchestrator_kwargs = {
            "n_iter_per_ss": new_n_iter_per_ss,
            "breaks": new_breaks,
            "time_before_exam": self.time_before_exam,
            "exam_threshold": self.exam_threshold,
            "inter_trial": self.inter_trial,
        }

        # ============ Create game_state to which the simulation will be reset to
        reset_dic = self.construct_reset_state_for_simu(
            copy.deepcopy(agent_observation)
        )  # Deepcopy just to be sure there is no interaction
        n_item = self.n_item
        task_class_kwargs = copy.deepcopy(self.task_class_kwargs)
        user_class_kwargs = copy.deepcopy(self.user_class_kwargs)
        # start simulation to check if myopic policy will lead to all items preented being learned
        while True:
            item_selected, rewards = self.simulate_myopic(
                n_item,
                reset_dic,
                task_class_kwargs,
                user_class_kwargs,
                orchestrator_kwargs,
            )
            if (
                int(np.sum(list(rewards.values()))) == n_item
            ):  # if all all items can be remembered
                break
            else:

                # make reduced model using only presented items
                n_item, reset_dic, task_class_kwargs, user_class_kwargs = self._reduce(
                    item_selected,
                    n_item,
                    reset_dic,
                    task_class_kwargs,
                    user_class_kwargs,
                )

        print(f"selected item {item_selected}")
        return item_selected, 0

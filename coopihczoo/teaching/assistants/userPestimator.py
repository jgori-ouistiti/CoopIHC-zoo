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
)
import numpy as np
import copy


class UserPEstimator(BaseAgent):
    def __init__(
        self, task_class, user_class, task_kwargs={}, user_kwargs={}, **kwargs
    ):

        inference_engine = DualInferenceEngine(
            primary_inference_engine=InferUserPRecall(),
            dual_inference_engine=BaseInferenceEngine(),
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
            name="Simulator >> UserPEstimator",
            task_model=self.task_model,
            user_model=self.user_model,
            assistant=self,
        )

    def finit(self, *args, **kwargs):

        n_item = self.n_item
        self.state["user_estimated_recall_probabilities"] = array_element(
            init=np.zeros((n_item,)), low=0, high=1, dtype=np.float64
        )

        # ================= Policy ============
        action_state = State()
        action_state["action"] = cat_element(N=n_item)

        agent_policy = BasePolicy(action_state=action_state)

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

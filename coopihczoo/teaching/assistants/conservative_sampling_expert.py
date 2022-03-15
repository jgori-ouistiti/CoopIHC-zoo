import numpy as np

from coopihc import BaseAgent, State, \
    cat_element, array_element, \
    RuleObservationEngine, oracle_engine_specification

from . conservative_sampling import ConservativeSamplingPolicy
from . rl import RlTeacherInferenceEngine


class ConservativeSamplingExpert(BaseAgent):

    def __init__(self, *args, **kwargs):
        super().__init__("assistant", *args, **kwargs)

    def finit(self, *args, **kwargs):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["progress"] = array_element(shape=1, low=0, high=np.inf, init=0.0)
        self.state["memory"] = array_element(shape=(n_item, 2), low=0, high=np.inf)

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(min=0, max=n_item)

        agent_policy = ConservativeSamplingPolicy(action_state=action_state)

        # Inference engine
        inference_engine = RlTeacherInferenceEngine()  # This is specific!!!

        # Use default observation engine
        observation_engine = RuleObservationEngine(
            deterministic_specification=oracle_engine_specification)

        self.attach_policy(agent_policy)
        self.attach_observation_engine(observation_engine)
        self.attach_inference_engine(inference_engine)

    def reset(self, dic=None):

        n_item = int(self.bundle.task.state["n_item"][0, 0])

        self.state["progress"][:] = 0
        self.state["memory"][:] = np.zeros((n_item, 2))

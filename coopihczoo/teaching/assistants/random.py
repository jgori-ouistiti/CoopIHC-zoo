from coopihc import BaseAgent, State, StateElement, \
    BasePolicy, autospace
import numpy as np


class RandomTeacher(BaseAgent):

    def __init__(self, n_item, *args, **kwargs):

        # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            0,
            autospace(np.arange(n_item))
        )
        agent_policy = BasePolicy(action_state=action_state)

        # Use default
        observation_engine = None
        inference_engine = None
        agent_state = None

        super().__init__(
            "assistant",
            *args,
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=agent_state,
            **kwargs
        )

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4

        :meta public:
        """
        pass

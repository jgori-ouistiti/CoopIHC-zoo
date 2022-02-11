from coopihc import BaseAgent, State, StateElement, Space, ExamplePolicy, BasePolicy, autospace
import numpy as np


class Assistant(BaseAgent):
    """An Example of a User.

    An agent that handles the ExamplePolicy, has a single 1d state, and has the default observation and inference engines.
    See the documentation of the :py:mod:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>` class for more details.

    :meta public:
    """

    def __init__(self, n_item, *args, **kwargs):

        # # Define an internal state with a 'goal' substate
        # state = State()
        # state["goal"] = StateElement(
        #     4,
        #     Space(
        #         np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int16),
        #         "discrete",
        #     ),
        # )
        #
        # # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            0,
            autospace(np.arange(n_item))
        )
        agent_policy = BasePolicy(action_state=action_state)
        agent_state = None

        # # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            *args,
            agent_policy=agent_policy,
            # policy_kwargs={"action_state": action_state},
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
        # self.state["goal"][:] = 4

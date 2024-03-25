from coopihc.agents.BaseAgent import BaseAgent

from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification

from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element

import numpy

from coopihc.policy import BasePolicy


class User(BaseAgent):
    """A user that do nothing.

    .. warning ::

        This agent only works with a task that has a 'targets' substate.


    * Reset: At each reset, it selects a new goal from the possible 'targets'.
    * Inference: None
    * State: None
    * Policy: When sampled, the user will issue an action that is either +1 or -1 in the direction of the target.
    * Observation: The user observes everything perfectly except for the assistant state.




    :param error_rate: rate at which users makes errors, defaults to 0.05
    :type error_rate: float, optional
    """

    def __init__(self, *args, error_rate=0.05, **kwargs):

        self._targets = None

        action_state = State()
        action_state["action"] = discrete_array_element(low=-1, high=1)
        state = State()

        #agent_policy = State(action=discrete_array_element(low=-1, high=1, init=0))
        agent_policy = None

        # ---------- Observation engine ------------
        observation_engine = RuleObservationEngine(
            deterministic_specification=base_user_engine_specification,
        )

        # ---------- Calling BaseAgent class -----------
        # Calling an agent, set as an user, which uses our previously defined observation engine and without an inference engine.

        super().__init__(
            "user",
            *args,
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            **kwargs,
        )

    
    def finit(self):
        self.state["goal"] = discrete_array_element(
            low=0, high=(self.bundle.task.gridsize - 1)
        )

    @property
    def targets(self):
        return self.bundle.task.state["targets"]

    def reset(self, dic=None):
        index = numpy.random.randint(0, self.targets.size)
        self.state["goal"] = discrete_array_element(
            init=self.targets[index],
            low=self.targets.space[index].low,
            high=self.targets.space[index].high - 1,
        )

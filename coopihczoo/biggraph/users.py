import coopihc
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.agents.lqrcontrollers.IHCT_LQGController import IHCT_LQGController

from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification

from coopihc.bundle.Bundle import Bundle

from coopihc.space.Space import Space
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import autospace

from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.policy.WrapAsPolicy import WrapAsPolicy
from coopihc.policy.ELLDiscretePolicy import BadlyDefinedLikelihoodError

from coopihc.interactiontask.ClassicControlTask import ClassicControlTask


import numpy
import copy


class A(BaseAgent):
    def __init__(self, *args, error_rate=0.05, **kwargs):

        action_state = State()
        action_state["action"] = StateElement(
            0, autospace(numpy.array([0, 1, 2, 3, 4])), out_of_bounds_mode="warning"
        )

        ELLD_dic = {"compute_likelihood_args": {"error_rate": error_rate}}
        ELLD_dic.update(kwargs.get("policy_kwargs", {}))

        agent_policy = ELLDiscretePolicy(
            action_state=action_state,
            **ELLD_dic,
        )

        def compute_likelihood(self, action, observation, *args, **kwargs):
            error_rate = kwargs.get("error_rate", 0)
            # convert actions and observations
            goal = observation["user_state"]["goal"]
            position = observation["task_state"]["position"]
            # Write down all possible cases (5)
            # (1) Goal to the right, positive action
            if goal > position and action > 0:
                return 1 - error_rate
            # (2) Goal to the right, negative action
            elif goal > position and action < 0:
                return error_rate
            # (3) Goal to the left, positive action
            if goal < position and action > 0:
                return error_rate
            # (4) Goal to the left, negative action
            elif goal < position and action < 0:
                return 1 - error_rate
            elif goal == position and action == 0:
                return 1
            elif goal == position and action != 0:
                return 0
            elif goal != position and action == 0:
                return 0
            else:
                raise RuntimeError(
                    "warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition"
                )

            # Attach likelihood function to the policy

        agent_policy.attach_likelihood_function(compute_likelihood)

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

    @property
    def targets(self):
        return self.bundle.task.state["targets"]

    def reset(self, dic=None):
        index = numpy.random.randint(0, self.targets.size)
        self.state["goal"] = StateElement(
            self.targets[index],
            self.targets.spaces[index],
            out_of_bounds_mode="warning",
        )
        # self.state["goal"] = numpy.random.choice(
        #     [tv.squeeze() for tv in self.targets]
        # )

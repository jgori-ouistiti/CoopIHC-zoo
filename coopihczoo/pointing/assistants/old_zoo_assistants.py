import coopihc
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BIGDiscretePolicy import BIGDiscretePolicy
from coopihc.inference.GoalInferenceWithUserPolicyGiven import (
    GoalInferenceWithUserPolicyGiven,
)
from coopihc.base.Space import Space
from coopihc.base.StateElement import StateElement
from coopihc.base.utils import autospace, discrete_space

import numpy
import copy


class ConstantCDGain(BaseAgent):
    """A Constant CD Gain transfer function.

    Here the assistant just picks a fixed modulation.

    :param gain: (float) constant CD gain.

    :meta public:
    """

    def __init__(self, gain):
        self.gain = gain

        super().__init__("assistant")

    def finit(self):

        action_space = autospace(
            numpy.full((1, self.bundle.task.dim), self.gain),
            numpy.full((1, self.bundle.task.dim), self.gain),
        )
        self.policy.action_state["action"] = StateElement(
            numpy.array([self.gain for i in range(self.bundle.task.dim)]).reshape(
                1, -1
            ),
            action_space,
            out_of_bonuds_mode="warning",
        )


class BIGGain(BaseAgent):
    def __init__(self):

        super().__init__(
            "assistant", agent_inference_engine=GoalInferenceWithUserPolicyGiven()  #
        )

    def finit(self):
        action_state = self.bundle.game_state["assistant_action"]
        action_state["action"] = StateElement(
            0,
            autospace([i for i in range(self.bundle.task.gridsize)]),
            out_of_bounds_mode="error",
        )
        user_policy_model = copy.deepcopy(self.bundle.user.policy)
        agent_policy = BIGDiscretePolicy(action_state, user_policy_model)
        self.attach_policy(agent_policy)
        self.inference_engine.attach_policy(user_policy_model)

        self.state["beliefs"] = StateElement(
            numpy.array(
                [
                    1 / self.bundle.task.number_of_targets
                    for i in range(self.bundle.task.number_of_targets)
                ]
            ).reshape(-1, 1),
            autospace(
                numpy.zeros((1, self.bundle.task.number_of_targets)),
                numpy.ones((1, self.bundle.task.number_of_targets)),
            ),
            out_of_bounds_mode="error",
        )

    def reset(self, dic=None):
        self.state["beliefs"][:] = numpy.array(
            [
                1 / self.bundle.task.number_of_targets
                for i in range(self.bundle.task.number_of_targets)
            ]
        ).reshape(1, -1)

        # change theta for inference engine
        set_theta = [
            {
                ("user_state", "goal"): StateElement(
                    t,
                    discrete_space(numpy.array(list(range(self.bundle.task.gridsize)))),
                )
            }
            for t in self.bundle.task.state["targets"]
        ]

        self.inference_engine.attach_set_theta(set_theta)
        self.policy.attach_set_theta(set_theta)

        def transition_function(assistant_action, observation):
            """What future observation will the user see due to assistant action"""
            # always do this
            observation["assistant_action"]["action"] = assistant_action
            # specific to BIGpointer
            observation["task_state"]["position"] = assistant_action

            return observation

        self.policy.attach_transition_function(transition_function)

    def render(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"
        try:
            axtask, axuser, axassistant = args
            self.inference_engine.render(axassistant, mode=mode)
        except ValueError:
            self.inference_engine.render(mode=mode)

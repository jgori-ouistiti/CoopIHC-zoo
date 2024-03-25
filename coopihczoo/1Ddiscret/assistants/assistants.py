import coopihc
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BIGDiscretePolicy import BIGDiscretePolicy
from coopihc.inference.GoalInferenceWithUserPolicyGiven import (
    GoalInferenceWithUserPolicyGiven,
)
from coopihc.base.Space import Space
from coopihc.base.StateElement import StateElement
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc import BasePolicy, State
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
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

        self.policy.action_state["action"] = array_element(
            init=self.gain,
            low=numpy.full((1, self.bundle.task.dim), self.gain),
            high=numpy.full((1, self.bundle.task.dim), self.gain),
        )


class BIGGain(BaseAgent):
    def __init__(self):

        super().__init__(
            "assistant",
            agent_inference_engine=GoalInferenceWithUserPolicyGiven(),
            agent_policy=BasePolicy(),
        )

        self.end_init = False

    def finit(self):

        # action_state = self.policy.action_state
        action_state = State()
        action_state["action"] = discrete_array_element(
            init=0, low=0, high=self.bundle.task.gridsize, out_of_bounds_mode="error"
        )

        user_policy_model = copy.deepcopy(self.bundle.user.policy)
        # user_policy_model = ELLDiscretePolicy(State(action=discrete_array_element(low=-1, high=1, init=0)))
        def compute_likelihood(self, action, observation):
            # convert actions and observations

            action = action
            goal = observation["user_state"]["goal"]
            position = observation["task_state"]["position"]
            print("goal "+ goal)
            print("GPA")
            print(goal, position, action)

            # Write down all possible cases (5)
            # (1) Goal to the right, positive action
            if goal > position and action > 0:
                return 0.99
            # (2) Goal to the right, negative action
            elif goal > position and action <= 0:
                return 0.005
            # (3) Goal to the left, positive action
            if goal < position and action >= 0:
                return 0.005
            # (4) Goal to the left, negative action
            elif goal < position and action < 0:
                return 0.99
            elif goal == position and action == 0:
                return 1
            elif goal == position and action != 0:
                return 0
            else:
                print(goal, position, action)
                raise RunTimeError(
                    "warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition"
                )


        # Attach likelihood function to the policy
        user_policy_model.attach_likelihood_function(compute_likelihood)

        agent_policy = BIGDiscretePolicy(action_state, user_policy_model)
        self._attach_policy(agent_policy)
        self.inference_engine._attach_policy(user_policy_model)

        self.state["beliefs"] = array_element(
            init=1 / self.bundle.task.number_of_targets,
            low=numpy.zeros((self.bundle.task.number_of_targets,)),
            high=numpy.ones((self.bundle.task.number_of_targets,)),
            out_of_bounds_mode="error",
        )

    def reset(self, dic=None):
        self.state["beliefs"][...] = numpy.array(
            [
                1 / self.bundle.task.number_of_targets
                for i in range(self.bundle.task.number_of_targets)
            ]
        )

        # change theta for inference engine
        set_theta = [
            {
                ("user_state", "goal"): discrete_array_element(
                    init=t, low=0, high=self.bundle.task.number_of_targets
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

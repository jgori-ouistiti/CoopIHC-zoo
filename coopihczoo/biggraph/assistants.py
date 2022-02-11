from coopihc import (
    BaseAgent,
    GoalInferenceWithUserPolicyGiven,
    BIGDiscretePolicy,
    State,
    StateElement,
    Space,
    autospace,
)

import copy
import numpy


class B(BaseAgent):
    def __init__(self, N=8):
        self.N = N

        super().__init__(
            "assistant", agent_inference_engine=GoalInferenceWithUserPolicyGiven()  #
        )

    def finit(self):
        action_state = self.bundle.game_state["assistant_action"]
        action_state["action"] = StateElement(
            0, autospace([i for i in range(2 ** self.N)]), out_of_bounds_mode="error"
        )
        user_policy_model = copy.deepcopy(self.bundle.user.policy)
        agent_policy = BIGDiscretePolicy(action_state, user_policy_model)
        self.attach_policy(agent_policy)
        self.inference_engine.attach_policy(user_policy_model)

        self.state["beliefs"] = StateElement(
            numpy.array([1 / self.N for i in range(self.N)]).reshape(-1, 1),
            autospace(
                numpy.zeros((1, self.N)),
                numpy.ones((1, self.N)),
            ),
            out_of_bounds_mode="error",
        )

    def reset(self, dic=None):
        self.state["beliefs"][:] = numpy.array(
            [1 / self.N for i in range(self.N)]
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

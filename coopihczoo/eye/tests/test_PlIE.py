from coopihc.space.Space import Space
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import autospace

from coopihczoo.eye.utils import ProvideLikelihoodInferenceEngine

import numpy

game_state = State(
    game_info=State(
        turn_index=StateElement(
            numpy.array([0]), autospace([0, 1, 2, 3]), out_of_bounds_mode="raw"
        ),
        round_index=StateElement(
            numpy.array([0]), autospace([0, 1]), out_of_bounds_mode="raw"
        ),
    ),
    task_state=State(
        target=StateElement(
            numpy.array([[-0.30282614]]),
            autospace([[[-1.0]], [[1.0]]]),
            out_of_bounds_mode="clip",
        ),
        fixation=StateElement(
            numpy.array([[0.0]]),
            autospace([[[-1.0]], [[1.0]]]),
            out_of_bounds_mode="clip",
        ),
    ),
    user_state=State(
        **{
            "belief-mu": StateElement(
                numpy.array([[0.0]]),
                autospace([[[-1.0]], [[1.0]]]),
                out_of_bounds_mode="warning",
            ),
            "belief-sigma": StateElement(
                numpy.array([[1000.0]]),
                autospace([[[-numpy.inf]], [[numpy.inf]]]),
                out_of_bounds_mode="warning",
            ),
            "y": StateElement(
                numpy.array([[0.0]]),
                autospace([[[-1.0]], [[1.0]]]),
                out_of_bounds_mode="warning",
            ),
            "Sigma_0": StateElement(
                numpy.array([[1000.0]]),
                autospace([[[-numpy.inf]], [[numpy.inf]]]),
                out_of_bounds_mode="warning",
            ),
        }
    ),
    assistant_state=State(),
    user_action=State(
        action=StateElement(
            numpy.array([[0.15020657]]),
            autospace([[[-1.0]], [[1.0]]]),
            out_of_bounds_mode="warning",
        )
    ),
    assistant_action=State(
        action=StateElement(
            numpy.array([1]), autospace([0, 1]), out_of_bounds_mode="warning"
        )
    ),
)

print(game_state)


class Test(ProvideLikelihoodInferenceEngine):
    def __init__(self, noise_level, observation, *args, **kwargs):
        class Host:
            pass

        super().__init__(noise_level, *args, **kwargs)
        self.host = Host()
        self.host.role = "user"
        self.buffer = [observation]


inference_engine = Test(0.5, game_state)
state, reward = inference_engine.infer()
print(state)

import numpy
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class ProvideLikelihoodInferenceEngine(BaseInferenceEngine):
    def __init__(self, noise_level, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):
        if self.host.role == "user":
            user_state = agent_observation["user_state"]
        else:
            user_state = agent_observation["assistant_state"]

        target, position = (
            agent_observation["task_state"]["target"],
            agent_observation["task_state"]["fixation"],
        )

        user_state["y"][...] = agent_observation["task_state"]["target"][...]
        user_state["Sigma_0"] = eccentric_noise(target, position, self.noise_level)
        return user_state, 0

    def render(self, mode="text", ax_user=None, ax_assistant=None, ax_task=None):
        pass


def eccentric_noise(target, position, sdn_level):
    """Eccentric noise definition

    * Compute the distance between the target and the current fixation.
    * Compute the angle between the radial component and the x component
    * Express the diagonal covariance matrix in the radial/tangential frame.
    * Rotate that covariance matrix with the rotation matrix P

    :param target: true target position
    :param position: current fixation
    :param sdn_level: signal dependent noise level

    :return: covariance matrix in the XY axis

    :meta public:
    """
    target, position = target.squeeze(), position.squeeze()
    if target.shape == (2,):
        eccentricity = numpy.sqrt(numpy.sum((target - position) ** 2))
        cosalpha = (target - position)[0] / eccentricity
        sinalpha = (target - position)[1] / eccentricity
        _sigma = sdn_level * eccentricity
        sigma = numpy.array([[_sigma, 0], [0, 3 * _sigma / 4]])
        P = numpy.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
        return P @ sigma @ P.T
    elif target.shape == ():
        eccentricity = numpy.sqrt(numpy.sum((target - position) ** 2))
        return numpy.array([sdn_level * eccentricity]).reshape(1, 1)
    else:
        raise NotImplementedError

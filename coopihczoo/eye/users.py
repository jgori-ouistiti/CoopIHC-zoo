from coopihc import BaseAgent, array_element, State

from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification

from coopihc.policy.LinearFeedback import LinearFeedback


from coopihc.inference.LinearGaussianContinuous import LinearGaussianContinuous
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine
from coopihc.inference.CascadedInferenceEngine import CascadedInferenceEngine

from .utils import ProvideLikelihoodInferenceEngine, eccentric_noise

from scipy.linalg import toeplitz
import numpy
import copy


class ChenEye(BaseAgent):
    """Model based on that of Chen, Xiuli, et al. "An Adaptive Model of Gaze-based Selection" Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021., --> with a real 2D implementation and a couple changes"""

    @staticmethod
    def eccentric_motor_noise(action, observation, oculomotornoise, *args, **kwargs):

        noise_covariance_matrix = eccentric_noise(
            action, observation["task_state"]["fixation"], oculomotornoise
        )
        noise = numpy.random.multivariate_normal(
            numpy.zeros(shape=action.reshape(-1).shape), noise_covariance_matrix
        )
        return noise, noise_covariance_matrix

    @staticmethod
    def eccentric_perceptual_noise(_obs, game_state, perceptualnoise, *args):
        target = game_state["task_state"]["target"]
        position = game_state["task_state"]["fixation"]
        # Def of eccentric_noise in utils
        Sigma = eccentric_noise(target, position, perceptualnoise)
        noise = numpy.random.multivariate_normal(
            numpy.zeros(shape=target.reshape(-1).shape), Sigma
        ).reshape(-1, 1)
        return _obs + noise  # noise clipped automatically by StateElement behavior

    def __init__(self, perceptualnoise, oculomotornoise, dimension=2, *args, **kwargs):
        self.dimension = dimension
        self.perceptualnoise = perceptualnoise
        self.oculomotornoise = oculomotornoise

        # ============= Define Policy
        action_state = State()
        action_state["action"] = array_element(
            low=-numpy.ones((dimension, 1), dtype=numpy.float32),
            high=numpy.ones((dimension, 1), dtype=numpy.float32),
        )

        def noise_function(action, observation, oculomotornoise):
            noise_obs = State()
            noise_obs["task_state"] = State()
            noise_obs["task_state"]["target"] = action
            noise_obs["task_state"]["fixation"] = observation["task_state"]["fixation"]
            noise = self.eccentric_motor_noise(action, noise_obs, oculomotornoise)[0]

            return action + noise.reshape((-1, 1))

        agent_policy = LinearFeedback(
            action_state,
            ("user_state", "belief-mu"),
            noise_function=noise_function,
            noise_func_args=(self.oculomotornoise,),
        )
        # ============ Define observation Engine
        extraprobabilisticrules = {
            ("task_state", "target"): (
                self.eccentric_perceptual_noise,
                (self.perceptualnoise,),
            )
        }

        observation_engine = RuleObservationEngine(
            deterministic_specification=base_user_engine_specification,
            extraprobabilisticrules=extraprobabilisticrules,
        )

        # =============== Define inference Engine

        first_inference_engine = ProvideLikelihoodInferenceEngine(perceptualnoise)
        second_inference_engine = LinearGaussianContinuous()
        inference_engine = CascadedInferenceEngine(
            [first_inference_engine, second_inference_engine]
        )

        # ============= Define State
        belief_mu = array_element(
            low=-numpy.ones((dimension, 1), dtype=numpy.float32),
            high=numpy.ones((dimension, 1), dtype=numpy.float32),
        )

        belief_sigma = array_element(
            low=-numpy.inf
            * numpy.ones((self.dimension, self.dimension), dtype=numpy.float32),
            high=numpy.inf
            * numpy.ones((self.dimension, self.dimension), dtype=numpy.float32),
            init=numpy.full((self.dimension, self.dimension), numpy.inf),
        )

        state = State()
        state["belief-mu"] = belief_mu
        state["belief-sigma"] = belief_sigma
        state["y"] = copy.deepcopy(belief_mu)
        state["Sigma_0"] = copy.deepcopy(belief_sigma)

        super().__init__(
            "user",
            agent_state=state,
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
        )

    def finit(self):
        pass

    def reset(self, dic=None):
        """Reset the fixation at the center (0;0), reset the prior belief

        :meta public:
        """

        # Initialize here the start position of the eye as well as initial uncertainty
        observation = State()
        observation["task_state"] = State()
        observation["task_state"]["target"] = self.bundle.task.state["target"]
        observation["task_state"]["fixation"] = self.bundle.task.state["fixation"]
        # Initialize with a huge Gaussian noise so that the first observation massively outweights the prior. Put more weight on the pure variance components to ensure that it will behave well.
        Sigma = toeplitz([1000] + [100 for i in range(self.dimension - 1)])
        self.state["belief-mu"] = numpy.array([0 for i in range(self.dimension)])
        self.state["belief-sigma"] = Sigma
        self.state["y"] = numpy.array([0 for i in range(self.dimension)])
        self.state["Sigma_0"] = Sigma

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


import gym
import numpy
import copy


class TwoDCarefulPointer(BaseAgent):
    """An user that only indicates the right direction, with a fixed amplitude.

    Works with a task that has a 'targets' substate. At each reset, it selects a new goal from the possible 'targets'. When sampled, the user will issue an action that is either +1 or -1 in the direction of the target.
    The user observes everything perfectly except for the assistant state.


    :meta public:
    """

    def __init__(self, **kwargs):

        # --------- Defining the agent's policy ----------

        agent_policy = kwargs.get("agent_policy")
        if agent_policy is None:

            # 0 = no action
            # 1 = up
            # 2 = left
            # 3 = down
            # 4 = right

            agent_policy = ELLDiscretePolicy(
                action_space=[coopihc.space.Discrete(5)],
                action_set=[[[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]],
                clipping_mode="clip",
            )

            def leftrightupdown_likelihood(self, action, observation):
                # Actions are in human values, i.e. they are not necessarily in range(0,N)
                # convert actions and observations
                action = action["values"][0]
                # first component selected
                goal = observation["user_state"]["goal"]["values"][0]
                position = observation["task_state"]["position"]["values"][0]

                if action == 0:
                    if (goal == position).all():
                        return 1
                    else:
                        return 0

                if goal[0] < position[0] and goal[1] < position[1]:
                    if action == 3 or action == 2:
                        return 0.99 / 2
                    else:
                        return 0.01 / 2

                elif goal[0] < position[0] and goal[1] > position[1]:
                    if action == 1 or action == 2:
                        return 0.99 / 2
                    else:
                        return 0.01 / 2

                elif goal[0] < position[0] and goal[1] == position[1]:
                    if action == 2:
                        return 0.99
                    else:
                        return 0.01 / 3

                elif goal[0] > position[0] and goal[1] < position[1]:
                    if action == 3 or action == 4:
                        return 0.99 / 2
                    else:
                        return 0.01 / 2

                elif goal[0] > position[0] and goal[1] > position[1]:
                    if action == 4 or action == 1:
                        return 0.99 / 2
                    else:
                        return 0.01 / 2

                elif goal[0] > position[0] and goal[1] == position[1]:
                    if action == 4:
                        return 0.99
                    else:
                        return 0.01 / 3

                elif goal[0] == position[0] and goal[1] < position[1]:
                    if action == 3:
                        return 0.99
                    else:
                        return 0.01 / 3

                elif goal[0] == position[0] and goal[1] > position[1]:
                    if action == 1:
                        return 0.99
                    else:
                        return 0.01 / 3

                elif goal[0] == position[0] and goal[1] == position[1]:
                    return 0

                else:
                    raise BadlyDefinedLikelihoodError(
                        "warning, unable to compute likelihood with action: {} and observation: {} You may have not covered all cases in the likelihood definition".format(
                            str(action), str(observation)
                        )
                    )

            # Attach likelihood function to the policy
            agent_policy.attach_likelihood_function(leftrightupdown_likelihood)

        # ---------- Observation engine ------------
        # High-level specification
        observation_engine = kwargs.get("observation_engine")

        if observation_engine is None:
            base_user_engine_specification = [
                ("turn_index", "all"),
                ("task_state", "all"),
                ("user_state", "all"),
                ("assistant_state", None),
                ("user_action", "all"),
                ("assistant_action", "all"),
            ]
            # Additional deterministic and probabilistic 'rules' that can be added to the engine: for example, to add noise to a component, or to target one component in particular
            extradeterministicrules = {}
            extraprobabilisticrules = {}
            observation_engine = RuleObservationEngine(
                deterministic_specification=base_user_engine_specification,
                extradeterministicrules=extradeterministicrules,
                extraprobabilisticrules=extraprobabilisticrules,
            )

        # ---------- Calling BaseAgent class -----------
        # Calling an agent, set as an user, which uses our previously defined observation engine and without an inference engine.

        super().__init__(
            "user",
            policy=agent_policy,
            observation_engine=observation_engine,
            inference_engine=None,
        )

    def finit(self):
        self.target_values = self.bundle.task.state["targets"]["values"]
        target_spaces = self.bundle.task.state["targets"]["spaces"]

        self.state["goal"] = StateElement(
            values=[None],
            spaces=copy.deepcopy(self.bundle.task.state["position"]["spaces"]),
            possible_values=[[None]],
        )

    def reset(self, dic=None):
        if dic is None:
            super().reset()

        self.target_values = self.bundle.task.state["targets"]["values"]
        selected_target = numpy.random.choice(len(self.target_values))
        self.state["goal"]["values"] = self.target_values[selected_target]

        if dic is not None:
            super().reset(dic=dic)

    def render(self, *args, mode="text"):
        if "text" in mode:
            print("\n")
            print("Goal")
            print(self.state["goal"]["values"])
            print("Action")
            print(self.policy.action["values"])
        else:
            raise NotImplementedError


class CarefulPointer(BaseAgent):
    """A user that only indicates the right direction, with a fixed amplitude, and with some error rate.

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
        action_state["action"] = StateElement(
            0, autospace(numpy.array([-1, 0, 1])), out_of_bounds_mode="warning"
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


class LQGPointer(BaseAgent):
    """Use this class as template to build your new agent.

    :param myparameter(type): explain the parameter here
    :return: what is returned
    :meta public:
    """

    def __init__(
        self,
        timestep=0.01,
        I=0.25,
        b=0.2,
        ta=0.03,
        te=0.04,
        F=numpy.diag([0, 0, 0, 0.001]),
        G=0.03 * numpy.diag([1, 1, 0, 0]),
        C=numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
        Gamma=numpy.array(0.08),
        D=numpy.array(
            [[0.01, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 0.05, 0], [0, 0, 0, 0]]
        ),
        Q=numpy.diag([1, 0.01, 0, 0]),
        R=numpy.array([[1e-4]]),
        U=numpy.diag([1, 0.1, 0.01, 0]),
        *args,
        **kwargs
    ):

        self.timestep = timestep
        self.I = I
        self.b = b
        self.ta = ta
        self.te = te
        a1 = b / (ta * te * I)
        a2 = 1 / (ta * te) + (1 / ta + 1 / te) * b / I
        a3 = b / I + 1 / ta + 1 / te
        bu = 1 / (ta * te * I)
        Ac = numpy.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, -a1, -a2, -a3]])
        Bc = numpy.array([[0, 0, 0, bu]]).reshape((-1, 1))

        # Define the bundle with LQG control
        action_task = ClassicControlTask(
            timestep, Ac, Bc, F=F, G=G, discrete_dynamics=False, noise="off"
        )
        action_user = IHCT_LQGController(
            "user", timestep, Q, R, U, C, Gamma, D, noise="on"
        )
        # Used to be this, below not tested
        # action_bundle = SinglePlayUserAuto(
        action_bundle = Bundle(
            action_task,
            action_user,
            onreset_deterministic_first_half_step=True,
            start_at_action=True,
        )

        # Wrap it up in the LQGPointerPolicy
        class LQGPointerPolicy(WrapAsPolicy):
            def __init__(self, action_bundle, *args, **kwargs):
                action_state = State()
                action_state["action"] = StateElement(
                    values=[None],
                    spaces=[
                        gym.spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(1, 1))
                    ],
                )
                super().__init__(action_bundle, action_state, *args, **kwargs)

            def sample(self):
                cursor = copy.copy(self.observation["task_state"]["position"])
                target = copy.copy(self.observation["user_state"]["goal"])
                # allow temporarily
                cursor.clipping_mode = "warning"
                target.clipping_mode = "warning"

                tmp_box = StateElement(
                    values=[None],
                    spaces=gym.spaces.Box(
                        -self.host.bundle.task.gridsize + 1,
                        self.host.bundle.task.gridsize - 1,
                        shape=(1,),
                    ),
                    possible_values=[[None]],
                    clipping_mode="warning",
                )

                cursor_box = StateElement(
                    values=[None],
                    spaces=gym.spaces.Box(-0.5, 0.5, shape=(1,)),
                    possible_values=[[None]],
                    clipping_mode="warning",
                )

                tmp_box["values"] = [
                    numpy.array(v) for v in (target - cursor)["values"]
                ]
                init_dist = tmp_box.cast(cursor_box)["values"][0]

                _reset_x = self.xmemory
                _reset_x[0] = init_dist
                _reset_x_hat = self.xhatmemory
                _reset_x_hat[0] = init_dist
                action_bundle.reset(
                    dic={
                        "task_state": {"x": _reset_x},
                        "user_state": {"xhat": _reset_x_hat},
                    }
                )

                total_reward = 0
                N = int(self.host.bundle.task.timestep / self.host.timestep)

                for i in range(N):
                    observation, sum_rewards, is_done, rewards = self.step()
                    total_reward += sum_rewards
                    if is_done:
                        break

                # Store state for next usage
                self.xmemory = observation["task_state"]["x"]["values"][0]
                self.xhatmemory = observation["user_state"]["xhat"]["values"][0]

                # Cast as delta in correct units
                cursor_box["values"] = -self.xmemory[0] + init_dist
                delta = cursor_box.cast(tmp_box)
                possible_values = [-30 + i for i in range(61)]
                value = possible_values.index(int(numpy.round(delta["values"][0])))
                action = StateElement(
                    values=value,
                    spaces=coopihc.space.Discrete(61),
                    possible_values=[possible_values],
                )

                return action, total_reward

            def reset(self):
                self.xmemory = numpy.array([[0.0], [0.0], [0.0], [0.0]])
                self.xhatmemory = numpy.array([[0.0], [0.0], [0.0], [0.0]])

        agent_policy = kwargs.get("agent_policy")
        if agent_policy is None:
            agent_policy = LQGPointerPolicy(action_bundle)

        observation_engine = kwargs.get("observation_engine")
        if observation_engine is None:
            observation_engine = RuleObservationEngine(base_user_engine_specification)
            # give observation engine

        super().__init__(
            "user",
            policy=agent_policy,
            observation_engine=observation_engine,
        )

    def finit(self):
        self.target_values = self.bundle.task.state["targets"]["values"]
        target_spaces = self.bundle.task.state["targets"]["spaces"]

        self.state["goal"] = StateElement(
            values=[None],
            spaces=[coopihc.space.Discrete(self.bundle.task.gridsize)],
            possible_values=[[None]],
        )

    def reset(self, dic=None):
        if dic is None:
            super().reset()

        self.target_values = self.bundle.task.state["targets"]["values"]
        self.state["goal"]["values"] = numpy.random.choice(self.target_values)

        if dic is not None:
            super().reset(dic=dic)

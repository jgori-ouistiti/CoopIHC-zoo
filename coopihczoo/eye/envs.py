import numpy
import math
import copy

from coopihc.interactiontask import InteractionTask
from coopihc.space import StateElement, Space

# import eye.noise


class ChenEyePointingTask(InteractionTask):
    """An environment to simulate eye fixations.

    This task is a simple box with a target randomly positioned. The goal of the environment is to steer the focal point of the eye on top of the target. It can be associated with a user model only, whose actions are the fixations --- which determines the new focal point of the eye.

    This environment is adapted from Chen, Xiuli, et al. : "An Adaptive Model of Gaze-based Selection" Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021."

    Changes to the original environment include the ability to have a 1 or 2 D environment, as well as a different stopping condition: the target is considered reached as soon as the fixation is within a distance <= threshold to the target.

    :param fitts_W: width of the target
    :type fitts_W: float
    :param fitts_D: distance between the initial fixation and the target
    :type fitts_D: float
    :param threshold: characteristic size of the focal area, defaults to 0.04
    :type threshold: float, optional
    :param dimension: dimension of the task (1 or 2), defaults to 2
    :type dimension: int, optional
    """

    # threshold => d = L alpha with L = 0.5m and alpha = 5°
    def __init__(self, fitts_W, fitts_D, threshold=0.04, dimension=2):
        super().__init__()

        self.threshold = threshold
        self.fitts_W = fitts_W
        self.fitts_D = fitts_D
        self.dimension = dimension
        self.parity = 1

        self.state["targets"] = StateElement(
            values=[numpy.array([0 for i in range(dimension)])],
            spaces=Space(
                [
                    -numpy.ones((dimension,), dtype=numpy.float32),
                    numpy.ones((dimension,), dtype=numpy.float32),
                ]
            ),
            clipping_mode="error",
        )

        self.state["fixation"] = StateElement(
            values=[numpy.array([0 for i in range(dimension)])],
            spaces=Space(
                [
                    -numpy.ones((dimension,), dtype=numpy.float32),
                    numpy.ones((dimension,), dtype=numpy.float32),
                ]
            ),
            clipping_mode="clip",
        )

    def get_new_target(self, D):

        if self.dimension == 2:
            angle = numpy.random.uniform(0, math.pi * 2)
            # d = numpy.random.uniform(-D/2,D/2)
            d = D / 2
            x_target = math.cos(angle) * d
            y_target = math.sin(angle) * d
            return numpy.array([x_target, y_target])
        elif self.dimension == 1:
            self.parity = (self.parity + 1) % 2
            # angle = self.parity*math.pi
            angle = numpy.random.binomial(1, 1 / 2) * math.pi
            d = numpy.random.uniform(-D / 2, D / 2)
            x_target = math.cos(angle) * d
            return numpy.array([x_target])
        else:
            raise NotImplementedError

    def reset(self, dic=None):
        """Resets the task

        Resets the fixation at the center of the box, and selects a random new target.

        :param dic: reset_dic (see the documentation of the reset mechanism in CoopIHC), defaults to None
        :type dic: dictionnary, optional
        """

        self.state["targets"]["values"] = numpy.array(
            [self.get_new_target(self.fitts_D)]
        )
        self.state["fixation"]["values"] = numpy.array(
            [0 for i in range(self.dimension)]
        )

    def _is_done_user(self):
        if (
            numpy.sqrt(
                numpy.sum(
                    (
                        self.state["fixation"]["values"]
                        - self.state["targets"]["values"][0]
                    )
                    ** 2
                )
            )
            - self.fitts_W / 2
            < self.threshold
        ):
            is_done = True
        else:
            is_done = False
        return is_done

    def user_step(self, user_action):
        """Task transition function on user action

        Move the focal point of the eye to the received action.

        :param user_action: new position of the focal point of the eye
        :type user_action: coopihc.space.StateElement object
        :return: tuple(task state, reward, isdone_flag, empty dictionnary)
        :rtype: tuple(coopihc.space.state, float, boolean)
        """
        self.state["fixation"]["values"] = copy.copy(user_action["values"])

        reward = -1

        return self.state, reward, self._is_done_user(), {}

    def assistant_step(self, assistant_action):
        return self.state, 0, False, {}

    def render(self, *args, mode="text", **kwargs):
        """Renders the task.

        If 'text' is in mode, then simply print out target and fixation positions.

        If 'plot' is in mode, it plots the fixations and the goals in the Box. If this mode is provided, you should pass (in that order) ``axtask, axuser, axassistant`` as args.

        :param mode: how to render, defaults to "text"
        :type mode: str, optional
        """

        goal = self.state["targets"]["values"][0].squeeze().tolist()
        fx = self.state["fixation"]["values"][0].squeeze().tolist()

        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.bundle.round_number))
            print("Target:")
            print(goal)
            print("fixation:")
            print(fx)

        if "plot" in mode:
            try:
                axtask, axuser, axassistant = args[:3]
            except ValueError:
                raise ValueError(
                    "You have to provide the three axes (task, user, assistant) to render in plot mode."
                )
            if self.ax is not None:
                pass
            else:
                self.ax = axtask
                self.ax.set_xlim([-1.3, 1.3])
                self.ax.set_ylim([-1.3, 1.3])
                self.ax.set_aspect("equal")

                axuser.set_xlim([-1.3, 1.3])
                axuser.set_ylim([-1.3, 1.3])
                axuser.set_aspect("equal")

            if self.dimension == 1:
                goal = [goal, 0]
                fx = [fx, 0]
            pgoal = self.ax.plot(*goal, "ro")
            traj = self.ax.plot(*fx, "og")
        if not ("plot" in mode or "text" in mode):
            raise NotImplementedError

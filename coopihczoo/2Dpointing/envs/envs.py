import numpy
import matplotlib.pyplot as plt
from collections import OrderedDict
from coopihc.interactiontask.InteractionTask import InteractionTask
from coopihc.interactiontask.PipeTaskWrapper import PipeTaskWrapper

PipeTaskWrapper
from coopihc.base.Space import Space
from coopihc.base.StateElement import StateElement
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.helpers import flatten
from coopihc.helpers import sort_two_lists
import functools

GRID_SIZE = (10, 10)
NUM_TARGET = 10

class DiscretePointingTaskPipeWrapper(PipeTaskWrapper):
    def __init__(self, task, pipe):
        init_message = {
            "number_of_targets": task.number_of_targets,
            "gridsize": task.gridsize,
            "goal": int(task.bundle.user.state.goal),
        }
        super().__init__(task, pipe, init_message)

    def update_task_state(self, state):
        # print(f"===============\n updating state, {state} \n old state {self.state}")
        for key in self.state:
            try:
                value = state.pop(key)
                if key == "position":
                    self.state[key] = numpy.array(value)
                elif key == "targets":
                    self.state[key] = [numpy.array(v) for v in value]
            except KeyError:
                raise KeyError(
                    'Key "{}" defined in task state was not found in the received data'.format(
                        key
                    )
                )
        if state:
            print(
                "warning: the received data has not been consumed. {} does not match any current task state".format(
                    str(state)
                )
            )
        # print(f"now: {self.state}")
            
    def update_user_state(self, state):
        for key in state:
            try:
                self.bundle.user.state[key]
            except KeyError:
                raise KeyError(
                    'Key "{}" sent in received data was not found in user state'.format(
                        key
                    )
                )
            self.bundle.user.state[key] = state[key]


class cell:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def toString(self):
        return "({}, {})".format(x, y)

class twoDPointingTask(InteractionTask):
    """A 2D pointing task.

    A 2D grid of size 'Gridsize'. The cursor is at a certain 'position' and there are several potential 'targets' on the grid. The user action is modulated by the assistant.

    :param gridsize: cell (int x int) Size of the grid
    :param number_of_targets: (int) Number of targets on the grid

    :meta public:
    """

    @property
    def user_action(self):
        return super().user_action[0]

    @property
    def assistant_action(self):
        return super().assistant_action[0]

    def on_bundle_constraints(self):
        if not hasattr(self.bundle.user.state, "goal"):
            raise AttributeError(
                "You must pair this task with a user that has a 'goal' state"
            )

    def __init__(self, gridsize=GRID_SIZE, number_of_targets=NUM_TARGET, mode="gain"):
        super().__init__()
        self.gridsize = gridsize
        self.numGrid = gridsize[0]*gridsize[1]
        self.number_of_targets = number_of_targets
        self.mode = mode
        self.dim = 1

        self.state["position"] = discrete_array_element(
            low=0, high=self.numGrid - 1, out_of_bounds_mode="clip"
        )

        self.state["targets"] = discrete_array_element(
            low=0, high=self.numGrid, shape=(number_of_targets,)
        )

    def reset(self, dic=None):
        """Reset the task.

        Reset the grid used for rendering, define new targets, select a starting position

        :param args: reset to the given state

        :meta public:
        """
        self.grid = [" " for i in range(self.numGrid)]
        targets = sorted(
            numpy.random.choice(
                list(range(self.numGrid)), size=self.number_of_targets, replace=False
            )
        )
        for i in targets:
            self.grid[i] = "T"
        # Define starting position
        copy = list(range(len(self.grid)))
        for i in targets:
            copy.remove(i)
        position = int(numpy.random.choice(copy))
        self.state["position"][...] = position
        self.state["targets"][...] = targets

    def on_user_action(self, *args, user_action=None, **kwargs):
        """Do nothing, increment turns, return half a timestep

        :meta public:
        """
        is_done = False
        if (
            # self.user_action == 0
            # and self.state["position"] == self.bundle.user.state["goal"]
            self.state["position"]
            == self.bundle.user.state["goal"]
        ):
            is_done = True
        return self.state, -1, is_done

    def on_assistant_action(self, *args, assistant_action=None, **kwargs):
        """Modulate the user's action.

        Multiply the user action with the assistant action.
        Update the position and grids.

        :param assistant_action: (list)

        :return: new state (OrderedDict), half a time step, is_done (True/False)

        :meta public:
        """
        is_done = False

        # Stopping condition if too many turns
        if int(self.round_number) >= 50:
            return self.state, 0, True

        if self.mode == "position":
            self.state["position"][...] = self.assistant_action
        elif self.mode == "gain":
            position = self.state["position"]

            # self.state["position"][...] = numpy.round(
            #     position + self.user_action * self.assistant_action
            # )
            self.state["position"] = numpy.round(
                position + self.user_action * assistant_action
            )

        return self.state, 0, False
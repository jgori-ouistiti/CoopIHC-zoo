from coopihc import InteractionTask, StateElement, discrete_space, autospace
import numpy as np


class Task(InteractionTask):
    """
    """

    def __init__(self, n_item, max_iter=10, *args, **kwargs):

        # Call super().__init__() beofre anything else, which initializes some useful attributes, including a State (self.state) for the task

        super().__init__(*args, **kwargs)

        self.max_iter = max_iter

        # Describe the state. Here it is a single item which takes value in [-4, -3, ..., 3, 4]. The StateElement has out_of_bounds_mode = clip, which means that values outside the range will automatically be clipped to fit the space.
        self.state["iteration"] = StateElement(
            0,
            autospace(np.zeros((1, 1)),
                      np.full((1, 1), np.inf)
                      )
        )

        self.state["item"] = StateElement(
            0,
            autospace(np.zeros((1, 1)),
                      np.full((1, 1), n_item)
                      )
        )

        self.state["timestamp"] = StateElement(
            0,
            autospace(np.zeros((1, 1)),
                      np.full((1, 1), np.inf)
                      )
        )

    def reset(self, dic=None):
        # Always start with state 'x' at 0
        self.state["item"][:] = 0
        self.state["iteration"][:] = 0
        self.state["timestamp"][:] = 0
        return

    def user_step(self, *args, **kwargs):
        # Modify the state in place, adding the user action
        is_done = False
        reward = 0
        if self.state["iteration"] == self.max_iter:
            is_done = True
        return self.state, reward, is_done

    def assistant_step(self, *args, **kwargs):
        is_done = False

        # timestamp =
        item = self.assistant_action

        self.state["item"][:] = item
        # self.state["timestamp"][:] = timestamp
        reward = 0
        return self.state, reward, is_done

    def render(self, *args, mode="text"):

        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.turn_number.squeeze().tolist()))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError("Only 'text' mode implemented for this task")
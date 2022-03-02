from coopihc import InteractionTask, num_element, array_element
import numpy as np


class Task(InteractionTask):
    """ """

    def __init__(self, n_item, inter_trial, max_iter, is_item_specific=False, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Parameters
        self.state["n_item"] = array_element(shape=1, low=0, high=np.inf, init=n_item)
        self.state["max_iter"] = array_element(shape=1, low=0, high=np.inf, init=max_iter)
        self.state["inter_trial"] = array_element(shape=1, low=0, high=np.inf, init=inter_trial)
        self.state["is_item_specific"] = array_element(shape=1, low=0, high=np.inf, init=is_item_specific)

        # Actual state
        self.state["iteration"] = array_element(shape=1, low=0, high=np.inf, init=0)
        self.state["item"] = array_element(shape=1, low=0, high=np.inf, init=0)
        self.state["timestamp"] = array_element(shape=1, low=0, high=np.inf, init=0)

    def reset(self, dic=None):

        self.state["item"][:] = -1
        self.state["iteration"][:] = -1
        self.state["timestamp"][:] = -1

    def user_step(self, *args, **kwargs):

        is_done = False
        reward = 0
        self.state["iteration"][:] += 1
        if self.state["iteration"] == self.state["max_iter"]:
            is_done = True
        return self.state, reward, is_done

    def assistant_step(self, *args, **kwargs):
        is_done = False

        item = self.assistant_action

        self.state["item"][:] = item
        self.state["timestamp"][:] += self.state["inter_trial"]
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

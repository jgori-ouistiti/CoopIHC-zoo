import numpy as np

from coopihc import InteractionTask, array_element


class Task(InteractionTask):
    """ """

    def __init__(self, thr, n_item, inter_trial, max_iter, is_item_specific=False, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Parameters
        self.state["n_item"] = array_element(shape=1, low=0, high=np.inf, init=n_item)
        self.state["max_iter"] = array_element(shape=1, low=0, high=np.inf, init=max_iter)
        self.state["inter_trial"] = array_element(shape=1, low=0, high=np.inf, init=inter_trial)
        self.state["is_item_specific"] = array_element(shape=1, low=0, high=np.inf, init=is_item_specific)

        self.state["log_thr"] = array_element(shape=1, low=-np.inf, high=np.inf, init=np.log(thr))

        # Actual state
        self.state["iteration"] = array_element(shape=1, low=0, high=np.inf, init=0)
        self.state["item"] = array_element(shape=1, low=0, high=np.inf, init=0)
        self.state["timestamp"] = array_element(shape=1, low=0, high=np.inf, init=0)

    def reset(self, dic=None):

        self.state["item"][:] = 0
        self.state["iteration"][:] = 0
        self.state["timestamp"][:] = 0

    def user_step(self, *args, **kwargs):

        # last_item = int(self.observation["task_state"]["item"])
        # last_time_reply = self.observation["task_state"]["timestamp"]
        #
        # last_was_success = self.observation["user_action"]["action"][0]

        current_iter = float(self.bundle.game_state["task_state"]["iteration"])
        max_iter = float(self.bundle.game_state["task_state"]["max_iter"])

        reward = 0
        if current_iter == max_iter - 1:
            init_forget_rate = self.bundle.game_state["user_state"]["param"][0, 0]
            rep_effect = self.bundle.game_state["user_state"]["param"][1, 0]

            n_pres = self.bundle.game_state["user_state"]["n_pres"].view(np.ndarray).flatten()
            last_pres = self.bundle.game_state["user_state"]["last_pres"].view(np.ndarray).flatten()

            log_thr = float(self.bundle.game_state["task_state"]["log_thr"])

            seen = n_pres > 0
            rep = n_pres[seen] - 1.  # only consider already seen items

            forget_rate = init_forget_rate * (1 - rep_effect) ** rep
            lop_p = -forget_rate * (current_iter - last_pres[seen])
            reward = np.sum(lop_p > log_thr)

            print("Final reward", reward)

        is_done = False
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

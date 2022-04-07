import numpy as np

from coopihc import InteractionTask, discrete_array_element, array_element


class Task(InteractionTask):
    """ """

    def __init__(self, thr, n_item, inter_trial, n_iter_per_ss,
                 n_session,
                 break_length, is_item_specific,
                 time_before_exam,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Parameters
        self.state["n_item"] = discrete_array_element(low=0, high=np.inf, init=n_item)
        self.state["n_iter_per_ss"] = discrete_array_element(low=0, high=np.inf, init=n_iter_per_ss)
        self.state["n_session"] = discrete_array_element(low=0, high=np.inf, init=n_session)
        self.state["inter_trial"] = discrete_array_element(low=0, high=np.inf, init=inter_trial)
        self.state["break_length"] = discrete_array_element(low=0, high=np.inf, init=break_length)  # 24*60**2
        self.state["is_item_specific"] = discrete_array_element(low=0, high=np.inf, init=is_item_specific)
        self.state["time_before_exam"] = discrete_array_element(low=0, high=np.inf, init=time_before_exam)

        self.state["log_thr"] = array_element(low=-np.inf, high=np.inf, init=np.log(thr))

        # Actual state
        self.state["iteration"] = discrete_array_element(low=0, high=np.inf)
        self.state["session"] = discrete_array_element(low=0, high=np.inf)
        self.state["item"] = discrete_array_element(low=0, high=np.inf)
        self.state["timestamp"] = discrete_array_element(low=0, high=np.inf)

    def reset(self, dic=None):

        self.state["item"] = 0
        self.state["iteration"] = 0
        self.state["timestamp"] = 0
        self.state["session"] = 0

    def on_user_action(self, *args, user_action=None, **kwargs):

        iteration = int(self.state["iteration"])
        n_iter_per_ss = int(self.state["n_iter_per_ss"])
        session = int(self.state["session"])
        n_session = int(self.state["n_session"])
        is_item_specific = bool(self.state["is_item_specific"])

        exam_timestamp = int(self.state["timestamp"]) + int(self.state["time_before_exam"])

        reward = 0
        if iteration == n_iter_per_ss - 1 and session == n_session - 1:

            n_pres = self.bundle.game_state["user_state"]["n_pres"].view(np.ndarray).flatten()
            last_pres = self.bundle.game_state["user_state"]["last_pres"].view(np.ndarray).flatten()

            log_thr = float(self.state["log_thr"])

            seen = n_pres > 0
            rep = n_pres[seen] - 1.  # only consider already seen items

            if is_item_specific:
                init_forget_rate = self.bundle.game_state["user_state"]["param"][seen, 0]
                rep_effect = self.bundle.game_state["user_state"]["param"][seen, 1]
            else:
                init_forget_rate = self.bundle.game_state["user_state"]["param"][0]
                rep_effect = self.bundle.game_state["user_state"]["param"][1]

            forget_rate = init_forget_rate * (1 - rep_effect) ** rep
            delta = exam_timestamp - last_pres[seen]
            lop_p = -forget_rate * delta
            reward = np.sum(lop_p > log_thr)

            # print("Final reward", reward)

        is_done = False

        self.state["iteration"] += 1
        if self.state["iteration"] == self.state["n_iter_per_ss"]:
            self.state["iteration"] = 0
            self.state["session"] += 1
            delta = int(self.state["break_length"])
        else:
            delta = int(self.state["inter_trial"])

        if self.state["session"] >= self.state["n_session"]:
            is_done = True

        self.state["timestamp"] += delta

        return self.state, reward, is_done

    def on_assistant_action(self, assistant_action=None, **kwargs):

        is_done = False
        reward = 0

        self.state["item"] = assistant_action

        return self.state, reward, is_done

    def render(self, *args, mode="text"):

        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.turn_number.squeeze().tolist()))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError("Only 'text' mode implemented for this task")

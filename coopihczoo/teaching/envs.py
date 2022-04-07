import numpy as np

from coopihc import InteractionTask, array_element
import copy


class Task(InteractionTask):
    """ """

    def __init__(
        self,
        thr,
        n_item,
        inter_trial,
        n_iter_per_ss,
        n_session,
        break_length,
        is_item_specific,
        time_before_exam,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        # Parameters
        self.parameters.update(
            {
                "n_item": n_item,
                "n_iter_per_ss": n_iter_per_ss,
                "n_session": n_session,
                "inter_trial": inter_trial,
                "break_length": break_length,
                "is_item_specific": is_item_specific,  # should be in user?
                "time_before_exam": time_before_exam,
                "log_thr": np.log(thr),
            }
        )

        # state
        self.state["iteration"] = array_element(low=0, high=np.inf)
        self.state["session"] = array_element(low=0, high=np.inf)
        self.state["item"] = array_element(low=0, high=np.inf)
        self.state["timestamp"] = array_element(low=0, high=np.inf)

    def reset(self, dic=None):

        self.state["item"] = 0
        self.state["iteration"] = 0
        self.state["timestamp"] = 0
        self.state["session"] = 0

    def on_user_action(self, *args, user_action=None, **kwargs):

        exam_timestamp = self.state["timestamp"] + self.time_before_exam

        iteration = self.state.iteration
        session = self.state.session

        reward = 0
        if iteration == self.n_iter_per_ss - 1 and session == self.n_session - 1:

            n_pres = self.bundle.game_state["user_state"]["n_pres"].view(np.ndarray)
            last_pres = self.bundle.game_state["user_state"]["last_pres"].view(
                np.ndarray
            )

            log_thr = self.log_thr

            seen = n_pres > 0
            rep = n_pres[seen] - 1.0  # only consider already seen items

            if self.is_item_specific:
                init_forget_rate = self.bundle.game_state["user_state"]["param"][
                    seen, 0
                ]
                rep_effect = self.bundle.game_state["user_state"]["param"][seen, 1]
            else:
                init_forget_rate = self.bundle.game_state["user_state"]["param"][0, 0]
                rep_effect = self.bundle.game_state["user_state"]["param"][1, 0]

            forget_rate = init_forget_rate * (1 - rep_effect) ** rep
            delta = exam_timestamp - last_pres[seen]
            lop_p = -forget_rate * delta
            reward = np.sum(lop_p > log_thr)

        is_done = False

        self.state["iteration"] += 1
        if self.state["iteration"] == self.n_iter_per_ss:
            self.state["iteration"] = 0
            self.state["session"] += 1
            delta = self.break_length
        else:
            delta = self.inter_trial

        if self.state["session"] >= self.n_session:
            is_done = True

        self.state["timestamp"] += delta

        return self.state, reward, is_done

    def on_assistant_action(self, *args, assistant_action=None, **kwargs):

        is_done = False
        reward = 0

        self.state["item"] = assistant_action

        return self.state, reward, is_done

    def render(self, *args, mode="text"):

        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.turn_number))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError("Only 'text' mode implemented for this task")

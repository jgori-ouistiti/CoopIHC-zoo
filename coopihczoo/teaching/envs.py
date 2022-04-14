import numpy as np
import copy

from coopihc import InteractionTask, discrete_array_element, Bundle

from coopihczoo.teaching.memory_models import ExponentialDecayMemory


class TeachingOrchestrator:
    def __init__(
        self,
        task=None,
        user=None,
        assistant=None,
        n_iter_per_ss=None,  # list of n_iter e.g. [10,20,20] (length N)
        breaks=None,  # list of break durations e.g. [30,20] (should be of length N-1 where)
        time_before_exam=None,
        exam_threshold=None,
        **kwargs,
    ):
        self.raw_bundle = Bundle(task=task, user=user, assistant=assistant, **kwargs)

        self.n_iter_per_ss = n_iter_per_ss
        self.breaks = breaks
        self.time_before_exam = time_before_exam
        self.exam_threshold = exam_threshold

        self.past_iter_per_ss_accumulator = -1
        self.break_number = 0

    # Make sure the bundle is not randomly reset
    def reset(self, **kwargs):
        kwargs.pop("random_reset", None)
        self.raw_bundle.reset(random_reset=False, **kwargs)

    def step(self, **kwargs):

        # ======================   If we have reached the last trial, compute the rewards

        # print(self.break_number)
        # print(len(self.n_iter_per_ss) - 1)
        # print(self.raw_bundle.round_number - self.past_iter_per_ss_accumulator)
        # print(self.n_iter_per_ss[self.break_number])

        if (
            self.break_number == len(self.n_iter_per_ss) - 1
            and self.raw_bundle.round_number - self.past_iter_per_ss_accumulator
            == self.n_iter_per_ss[self.break_number]
        ):
            state, _, _ = self.raw_bundle.step(**kwargs)
            p = state.user_state.recall_probabilities
            _reward = int(np.sum(p > self.exam_threshold))

            rewards = {}
            rewards["user_observation_reward"] = 0
            rewards["user_inference_reward"] = 0
            rewards["user_policy_reward"] = 0
            rewards["first_task_reward"] = _reward
            rewards["assistant_observation_reward"] = 0
            rewards["assistant_inference_reward"] = 0
            rewards["assistant_policy_reward"] = 0
            rewards["second_task_reward"] = 0
            return state, rewards, True

        # ======================  If we are changing sessions, increment time since last presentation to account for the break during sessions before playing out the Bundle.
        if (
            self.raw_bundle.round_number - self.past_iter_per_ss_accumulator
            == self.n_iter_per_ss[self.break_number]
        ):

            # Apply break (copy likely not needed, but let's be safe)
            game_state = copy.deepcopy(self.raw_bundle.game_state.filter(mode="array"))
            game_state["task_state"]["timestamp"] += self.breaks[self.break_number]

            self.reset(dic=game_state)

            self.past_iter_per_ss_accumulator += self.n_iter_per_ss[self.break_number]
            self.break_number += 1

        # Play out the bundle
        return self.raw_bundle.step(**kwargs)


class TaskWithoutSequence(InteractionTask):
    """ """

    def __init__(
        self, thr=None, n_item=None, inter_trial=None, is_item_specific=None, **kwargs
    ):

        super().__init__(**kwargs)

        # Parameters
        self.parameters.update(
            {
                "n_item": n_item,
                "inter_trial": inter_trial,
                "is_item_specific": is_item_specific,  # should be in user?
                "log_thr": np.log(thr),
            }
        )

        # state
        self.state["iteration"] = discrete_array_element(low=0, high=np.inf)
        self.state["item"] = discrete_array_element(low=0, high=np.inf)
        self.state["timestamp"] = discrete_array_element(low=0, high=np.inf)

    def reset(self, dic=None):
        self.state["item"] = 0
        self.state["iteration"] = 0
        self.state["timestamp"] = 0

    def on_user_action(self, *args, user_action=None, **kwargs):

        reward = 0
        is_done = False
        self.state["timestamp"] += self.inter_trial

        return self.state, reward, is_done

    def on_assistant_action(self, assistant_action=None, **kwargs):

        is_done = False
        reward = 0
        self.state["item"] = int(assistant_action)

        return self.state, reward, is_done


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
        **kwargs,
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

        exam_timestamp = self.state["timestamp"] + self.time_before_exam

        iteration = self.state.iteration
        session = self.state.session

        reward = 0
        if iteration == self.n_iter_per_ss - 1 and session == self.n_session - 1:

            n_pres = self.bundle.game_state["user_state"]["n_pres"]
            last_pres = self.bundle.game_state["user_state"]["last_pres"]

            log_thr = self.log_thr

            seen = n_pres > 0
            rep = n_pres[seen] - 1.0  # only consider already seen items

            if self.is_item_specific:
                init_forget_rate = self.bundle.game_state["user_state"]["param"][
                    seen, 0
                ]
                rep_effect = self.bundle.game_state["user_state"]["param"][seen, 1]
            else:
                init_forget_rate = self.bundle.game_state["user_state"]["param"][0]
                rep_effect = self.bundle.game_state["user_state"]["param"][1]

            forget_rate = init_forget_rate * (1 - rep_effect) ** rep
            delta = exam_timestamp - last_pres[seen]
            lop_p = -forget_rate * delta

            reward = int(np.sum(lop_p > log_thr))

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

    def on_assistant_action(self, assistant_action=None, **kwargs):

        is_done = False
        reward = 0

        self.state["item"] = int(assistant_action)

        return self.state, reward, is_done

    def render(self, *args, mode="text"):

        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.turn_number))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError("Only 'text' mode implemented for this task")

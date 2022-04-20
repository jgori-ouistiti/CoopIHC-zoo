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

            self.reset(dic=game_state, assistant_components="policy-observation")

            self.past_iter_per_ss_accumulator += self.n_iter_per_ss[self.break_number]
            self.break_number += 1

        # Play out the bundle
        return self.raw_bundle.step(**kwargs)


class TeachingTask(InteractionTask):
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
        self.state["item"] = discrete_array_element(low=0, high=np.inf)
        self.state["timestamp"] = discrete_array_element(low=0, high=np.inf)
        self.state["n_pres"] = discrete_array_element(
            shape=(n_item,), low=-1, high=np.inf
        )
        self.state["last_pres"] = discrete_array_element(
            shape=(n_item,), low=-np.inf, high=np.inf
        )

    def reset(self, dic=None):
        n_item = self.parameters["n_item"]
        self.state["item"] = 0
        self.state["timestamp"] = 0
        self.state["n_pres"] = np.zeros(n_item)
        self.state["last_pres"] = np.full((n_item,), -np.inf)

    def on_user_action(self, *args, user_action=None, **kwargs):

        reward = 0
        is_done = False
        self.state["timestamp"] += self.inter_trial

        return self.state, reward, is_done

    def on_assistant_action(self, assistant_action=None, **kwargs):

        is_done = False
        reward = 0
        item = int(assistant_action)
        self.state["item"] = item
        self.state["n_pres"][item] += 1
        self.state["last_pres"][item] = self.state["timestamp"][...]

        return self.state, reward, is_done

import numpy as np
import copy

from coopihc import InteractionTask, discrete_array_element, Bundle

# from coopihczoo.teaching.users.memory_models.exponential_decay import ExponentialDecayMemory


class TeachingOrchestrator:
    def __init__(
        self,
        object,
        n_iter_per_ss=None,  # list of n_iter e.g. [10,20,20] (length N)
        breaks=None,  # list of break durations e.g. [30,20] (should be of length N-1 where)
        time_before_exam=None,
        exam_threshold=None,
        inter_trial=None,
    ):
        self.raw_bundle = object

        self.inter_trial = inter_trial
        self.n_iter_per_ss = n_iter_per_ss
        self.breaks = breaks
        self.time_before_exam = time_before_exam
        self.exam_threshold = exam_threshold

        self.init_round = 0
        self.past_iter_per_ss_accumulator = -1
        self.break_number = 0
        self.counter = 0

    @staticmethod
    def reduce_schedule(n_iter_per_ss, breaks, current_iteration):
        iterations_in_schedule = np.cumsum(n_iter_per_ss).tolist()
        _appended_iterations_in_schedule = iterations_in_schedule + [current_iteration]
        index = sorted(_appended_iterations_in_schedule).index(current_iteration)
        new_n_iter_per_ss = [
            iterations_in_schedule[index] - current_iteration
        ] + n_iter_per_ss[index + 1 :]
        new_breaks = breaks[index:]
        return new_n_iter_per_ss, new_breaks

    # Make sure the bundle is not randomly reset
    def reset(self, **kwargs):
        kwargs.pop("random_reset", None)
        self.raw_bundle.reset(random_reset=False, **kwargs)

    def step(self, **kwargs):

        if self.breaks == [] and self.n_iter_per_ss == [0]:
            p = self.raw_bundle.state.user_state.recall_probabilities

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
            return self.raw_bundle.state, rewards, True

        if self.break_number == len(self.n_iter_per_ss) - 1 and (
            self.counter - self.past_iter_per_ss_accumulator
            == self.n_iter_per_ss[self.break_number]
        ):  # or (len(self.breaks) == 0 and self.n_iter_per_ss == [1]):
            state, _, _ = self.raw_bundle.step(**kwargs)  # last round
            self.counter += 1
            # do exam -- age the user obs
            last_user_obs = copy.deepcopy(self.raw_bundle.user.observation)
            last_user_obs["task_state"]["timestamp"] += self.time_before_exam

            # reset bundle to aged obs
            reset_dic = copy.deepcopy(state)
            reset_dic["task_state"]["timestamp"] += self.time_before_exam
            reset_dic["user_observation"] = last_user_obs
            self.raw_bundle.reset(
                dic=reset_dic, user_components="none", assistant_components="none"
            )

            # produce inference (decay user probabilities) and use for rewards
            self.raw_bundle.user.infer()
            p = self.raw_bundle.state.user_state.recall_probabilities

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

        # ======================  If we are changing sessions, increment time since last presentation to account for the break during sessions before playing out the Bundle. Also account for n_iter_per_ss = 0
        if (
            self.counter - self.past_iter_per_ss_accumulator
            == self.n_iter_per_ss[self.break_number]
        ) or (self.n_iter_per_ss[self.break_number] == 0):

            # Apply break (copy likely not needed, but let's be safe). Don't forget to substract an inter trial to the break.
            game_state = copy.deepcopy(self.raw_bundle.game_state.filter(mode="array"))
            game_state["task_state"]["timestamp"] += (
                self.breaks[self.break_number] - self.inter_trial
            )

            self.reset(dic=game_state, assistant_components="policy-observation")

            self.past_iter_per_ss_accumulator += self.n_iter_per_ss[self.break_number]
            self.break_number += 1

            if self.n_iter_per_ss[self.break_number] == 0:
                print("yes")
                return None, None, None

        # Play out the bundle
        self.counter += 1
        return self.raw_bundle.step(**kwargs)


class TeachingTask(InteractionTask):
    """ """

    def __init__(self, thr=None, n_item=None, inter_trial=None, **kwargs):

        super().__init__(**kwargs)

        # Parameters
        self.parameters.update(
            {
                "n_item": n_item,
                "inter_trial": inter_trial,
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
        # self.state["timestamp"] += self.inter_trial

        return self.state, reward, is_done

    def on_assistant_action(self, assistant_action=None, **kwargs):

        is_done = False
        reward = 0
        item = int(assistant_action)
        self.state["item"] = item
        self.state["n_pres"][item] += 1
        self.state["timestamp"] += self.inter_trial
        self.state["last_pres"][item] = self.state["timestamp"][...]

        return self.state, reward, is_done

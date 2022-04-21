from coopihczoo.teaching.users import ExponentialUser
from coopihczoo.teaching.envs import TeachingTask, TeachingOrchestrator
from coopihczoo.teaching.config import config_example
from coopihczoo.teaching.assistants.conservative_sampling import ConservativeSampling

from coopihczoo.teaching.config import config_example

from coopihc import Bundle, State, BufferNotFilledError, Simulator

import copy

import numpy as np

# Define a task
task = TeachingTask(**config_example.task_kwargs)
# Define a user
user = ExponentialUser(**config_example.user_per_item_kwargs)
user_model = ExponentialUser(**config_example.user_per_item_kwargs)
# Define an assistant
assistant = ConservativeSampling(
    TeachingTask,
    ExponentialUser,
    copy.deepcopy(config_example.teaching_orchestrator_kwargs),
    task_kwargs=config_example.task_kwargs,
    user_kwargs=config_example.user_per_item_kwargs,
)

bundle = Bundle(
    task=task, user=user, assistant=assistant, random_reset=False, seed=1234
)
bundle.reset(start_after=2, go_to=3)
print(bundle.state)
bundle.step()
print(bundle.state)
self = bundle.assistant.policy.primary_policy
agent_observation = self.observation
current_iteration = int(agent_observation.game_info.round_index)
# ============   Creating orchestrator schedule
# use the original schedule, and modify it by removing the current iteration from it (respecting breaks as well)
iterations_in_schedule = np.cumsum(self.n_iter_per_ss).tolist()
_appended_iterations_in_schedule = iterations_in_schedule + [current_iteration]
index = sorted(_appended_iterations_in_schedule).index(current_iteration)
new_n_iter_per_ss = [
    iterations_in_schedule[index] - current_iteration
] + self.n_iter_per_ss[index + 1 :]
new_breaks = self.breaks[index:]

orchestrator_kwargs = {
    "n_iter_per_ss": new_n_iter_per_ss,
    "breaks": new_breaks,
    "time_before_exam": self.time_before_exam,
    "exam_threshold": self.exam_threshold,
}
# ============ Create game_state to which the game will be reset the first time
game_reset_state = copy.deepcopy(
    agent_observation
)  # Deepcopy just to be sure there is no interaction

# Since user state is not observable, recreate it here from the assistant's knowledge ================
game_reset_state["user_state"] = State()
user_probs = game_reset_state.pop("assistant_state").pop(
    "user_estimated_recall_probabilities"
)
game_reset_state["user_state"]["recall_probabilities"] = user_probs
try:
    last_item = int(agent_observation["task_state"]["item"])
    past_observation = self.host.inference_engine.buffer[-2]
    user_last_pres_before_obs = past_observation["task_state"]["last_pres"][last_item]
    user_n_pres_before_obs = past_observation["task_state"]["n_pres"][last_item]
except BufferNotFilledError:  # Deal with start edge case
    user_n_pres_before_obs = 0
    user_last_pres_before_obs = 0

game_reset_state["user_state"]["n_pres_before_obs"] = user_n_pres_before_obs
game_reset_state["user_state"]["last_pres_before_obs"] = user_last_pres_before_obs
# ============================= End recreating user state


# =============== Init for conservative sampling
new_task_class_kwargs = copy.deepcopy(self.task_class_kwargs)
new_user_class_kwargs = copy.deepcopy(self.user_class_kwargs)
n_item = self.n_item
while True:
    simulator = Simulator(
        task_model=self.task_class(**new_task_class_kwargs),
        user_model=self.user_class(**new_user_class_kwargs),
        assistant=self.host,
        use_primary_inference=False,
        seed=1234,
        random_reset=False,
    )
    simulator.open()
    orchestrator = TeachingOrchestrator(simulator, **orchestrator_kwargs)
    orchestrator.reset(dic=copy.deepcopy(game_reset_state))
    if (
        orchestrator.raw_bundle.assistant.policy.mode != "dual"
        and orchestrator.raw_bundle.assistant.policy.dual_policy.__class__.__name__
        == "MyopicPolicy"
    ):
        raise RuntimeError(
            f"The orchestrator is not using the correct policy. Should be in dual mode with MyopicPolicy, but it is in {orchestrator.raw_bundle.assistant.policy.mode} mode instead"
        )
    k = 0
    while True:
        k += 1
        state, rewards, is_done = orchestrator.step()
        if k == 1:  # Remember which item was chosen as first item
            item_selected = state["assistant_action"]["action"]
        if is_done:
            break
    if int(np.sum(list(rewards.values()))) == n_item:
        print("conservative sampling converged")
        print(state)
        print(int(np.sum(list(rewards.values()))))
        break
exit()

# Bundle them together
orchestrator = TeachingOrchestrator(
    Bundle(
        task=task,
        user=user,
        assistant=assistant,
        random_reset=False,
        seed=1234,
    ),
    **config_example.teaching_orchestrator_kwargs,
)
orchestrator.reset(start_after=2, go_to=3)
while True:
    state, rewards, is_done = orchestrator.step()
    if is_done:
        break
print(orchestrator.raw_bundle.state)
exit()


def run_conservative():

    # Define a task
    task = TeachingTask(**config_example.task_kwargs)
    # Define a user
    user = ExponentialUser(**config_example.user_per_item_kwargs)
    user_model = ExponentialUser(**config_example.user_per_item_kwargs)
    # Define an assistant
    assistant = ConservativeSampling(
        task_class=TeachingTask,
        user_class=ExponentialUser,
        task_kwargs=config_example.task_kwargs,
        user_kwargs=config_example.user_per_item_kwargs,
    )
    # Bundle them together

    orchestrator = TeachingOrchestrator(
        task=task,
        user=user,
        assistant=assistant,
        random_reset=False,
        seed=1234,
        **config_example.teaching_orchestrator_kwargs,
    )
    orchestrator.reset(start_after=2, go_to=3)


if __name__ == "__main__":
    run_conservative()

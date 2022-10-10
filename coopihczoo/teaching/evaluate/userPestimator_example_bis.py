from coopihczoo.teaching.users.users import ExponentialUser
from coopihczoo.teaching.envs.envs import TeachingTask, TeachingOrchestrator
from coopihczoo.teaching.assistants.userPestimator import UserPEstimator

from coopihczoo.teaching.config import config_example

from coopihc import Bundle
import numpy
import copy


n_item = 20

# Some arguments are only for the evaluate_naive_implementation implementation
task_kwargs = dict(
    n_item=20,
    inter_trial=3,
    thr=0.9,
)


user_per_item_kwargs = dict(
    param=numpy.concatenate(
        (numpy.full((n_item, 1), 0.01), numpy.full((n_item, 1), 0.2)), axis=1
    )
)

teaching_orchestrator_kwargs = dict(
    n_iter_per_ss=[100, 100],
    breaks=[30],
    time_before_exam=0,
    inter_trial=3,
    exam_threshold=0.9,
)


def run_userpestimator():

    task = TeachingTask(**task_kwargs)
    # Define a user
    user = ExponentialUser(**user_per_item_kwargs)
    # Define an assistant
    assistant = UserPEstimator(
        task_class=TeachingTask,
        user_class=ExponentialUser,
        task_kwargs=task_kwargs,
        user_kwargs=user_per_item_kwargs,
    )
    # Bundle them together
    orchestrator = TeachingOrchestrator(
        Bundle(
            task=task,
            user=user,
            assistant=assistant,
            random_reset=False,
            seed=1234,
        ),
        **teaching_orchestrator_kwargs,
    )
    orchestrator.reset(start_after=2, go_to=3)
    while True:
        state, rewards, is_done = orchestrator.step()
        if is_done:
            break

    print(f"reward: {int(numpy.sum(list(rewards.values())))}")
    print(state)


def run_userpestimator_mismatch():

    task = TeachingTask(**task_kwargs)
    # Define a user
    user = ExponentialUser(**user_per_item_kwargs)
    # Define an assistant

    mismatch = copy.deepcopy(user_per_item_kwargs)
    mismatch["param"] += 0.1

    assistant = UserPEstimator(
        task_class=TeachingTask,
        user_class=ExponentialUser,
        task_kwargs=config_example.task_kwargs,
        user_kwargs=mismatch,
    )
    # Bundle them together
    orchestrator = TeachingOrchestrator(
        Bundle(
            task=task,
            user=user,
            assistant=assistant,
            random_reset=False,
            seed=1234,
        ),
        **teaching_orchestrator_kwargs,
    )
    orchestrator.reset(start_after=2, go_to=3)
    while True:
        state, rewards, is_done = orchestrator.step()
        if is_done:
            break

    print(f"reward: {int(numpy.sum(list(rewards.values())))}")
    print(state)


if __name__ == "__main__":
    run_userpestimator()
    run_userpestimator_mismatch()

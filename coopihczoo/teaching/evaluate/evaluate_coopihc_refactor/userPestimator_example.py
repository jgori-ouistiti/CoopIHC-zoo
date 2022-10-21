from coopihczoo.teaching.users.policy import ExponentialUser
from coopihczoo.teaching.envs.envs import TeachingTask, TeachingOrchestrator
from coopihczoo.teaching.assistants.userPestimator import UserPEstimator

from coopihczoo.teaching.config import config_example

from coopihc import Bundle
import numpy
import copy


def run_userpestimator():

    task = TeachingTask(**config_example.task_kwargs)
    # Define a user
    user = ExponentialUser(**config_example.user_per_item_kwargs)
    # Define an assistant
    assistant = UserPEstimator(
        task_class=TeachingTask,
        user_class=ExponentialUser,
        task_kwargs=config_example.task_kwargs,
        user_kwargs=config_example.user_per_item_kwargs,
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
        **config_example.teaching_orchestrator_kwargs,
    )
    orchestrator.reset(start_after=2, go_to=3)
    while True:
        state, rewards, is_done = orchestrator.step()
        if is_done:
            break

    print(f"reward: {int(numpy.sum(list(rewards.values())))}")
    print(state)


def run_userpestimator_mismatch():

    task = TeachingTask(**config_example.task_kwargs)
    # Define a user
    user = ExponentialUser(**config_example.user_per_item_kwargs)
    # Define an assistant

    mismatch = copy.deepcopy(config_example.user_per_item_kwargs)
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
        **config_example.teaching_orchestrator_kwargs,
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

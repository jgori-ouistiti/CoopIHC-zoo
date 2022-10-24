from coopihczoo.teaching.users.users import ExponentialUser
from coopihczoo.teaching.envs.envs import TeachingTask, TeachingOrchestrator
from coopihczoo.teaching.assistants.conservative_sampling import ConservativeSampling

from coopihczoo.teaching.config import config_example

from coopihc import Bundle

import copy

import numpy as np


def run_conservative():

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
    j = 0
    while True:
        print("Step", j)
        j += 1
        state, rewards, is_done = orchestrator.step()
        print( state)
        exit()
        if is_done:
            break

    print(f"reward: {int(np.sum(list(rewards.values())))}")
    print(state)


if __name__ == "__main__":
    run_conservative()

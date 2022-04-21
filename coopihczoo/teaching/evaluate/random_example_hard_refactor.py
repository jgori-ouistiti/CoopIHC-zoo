from coopihc import Bundle

from coopihczoo.teaching.users import ExponentialUser
from coopihczoo.teaching.envs import TeachingTask, TeachingOrchestrator
from coopihczoo.teaching.assistants.random import RandomTeacher
from coopihczoo.teaching.config import config_example

import numpy


def run_random():

    task = TeachingTask(**config_example.task_kwargs)
    user = ExponentialUser(**config_example.user_per_item_kwargs)
    assistant = RandomTeacher()

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
    run_random()

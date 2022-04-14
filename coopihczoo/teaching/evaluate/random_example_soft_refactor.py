from coopihc import Bundle

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task, TaskWithoutSequence, TeachingOrchestrator
from coopihczoo.teaching.assistants.random import RandomTeacher
from coopihczoo.teaching.config import config_example

import numpy


def run_random():

    task = Task(**config_example.task_kwargs)
    user = User(**config_example.user_kwargs)
    assistant = RandomTeacher()
    bundle = Bundle(
        task=task, user=user, assistant=assistant, random_reset=False, seed=1234
    )
    bundle.reset(
        start_after=2,
        go_to=3,
    )

    while 1:
        state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
        if is_done:
            break

    print("Final reward", rewards["first_task_reward"])
    print(state)


if __name__ == "__main__":
    run_random()

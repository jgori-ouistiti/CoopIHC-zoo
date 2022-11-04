import numpy

from coopihc import Bundle

from coopihczoo.teaching.users.memory_models.act_r import ActRUser
from coopihczoo.teaching.envs.envs import TeachingTask, TeachingOrchestrator
from coopihczoo.teaching.assistants.assistants_coopihc_refactor.random import (
    RandomTeacher,
)
from coopihczoo.teaching.config import config_example


def run_random():

    task = TeachingTask(**config_example.task_kwargs)
    user = ActRUser(**config_example.user_act_r_kwargs)
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
    task = TeachingTask(**config_example.task_kwargs)
    user = ActRUser(**config_example.user_act_r_kwargs)
    assistant = RandomTeacher()

    bundle = Bundle(
        task=task,
        user=user,
        assistant=assistant,
        random_reset=False,
        seed=1234,
    )
    bundle.reset(start_after=2, go_to=3)

    run_random()

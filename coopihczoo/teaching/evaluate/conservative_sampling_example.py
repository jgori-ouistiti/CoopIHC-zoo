from coopihczoo.teaching.users import ExponentialUser
from coopihczoo.teaching.envs import TeachingTask, TeachingOrchestrator
from coopihczoo.teaching.config import config_example
from coopihczoo.teaching.assistants.conservative_sampling import ConservativeSampling

from coopihczoo.teaching.config import config_example

from coopihc import Bundle

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

from coopihc import Bundle

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.conservative_sampling import ConservativeSampling

from coopihczoo.teaching.config import config_example


def run_conservative():

    task = Task(**config_example.task_kwargs)
    user = User(**config_example.user_kwargs)
    assistant = ConservativeSampling()
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False)
    # Reset the bundle (i.e. initialize it to a random or prescribed states)
    ## 0 : after assistant takes action + new task state
    ## 1 : after user observation + user inference + new user state
    ## 2 : after user takes action + new task state
    ## 3 : after assistant observation + assitant inference
    bundle.reset(
        start_at=3,
        go_to=3
    )  # Reset in a state where the user has already produced an observation and made an inference.
    while True:
        state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
        if is_done:
            break

    print("Final reward", rewards['first_task_reward'])


if __name__ == "__main__":
    run_conservative()

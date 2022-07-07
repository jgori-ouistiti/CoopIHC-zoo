from coopihc import Bundle

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.leitner import Leitner

from coopihczoo.teaching.config import config_example


def run_leitner():

    # Define a task
    task = Task(**config_example.task_kwargs)
    # Define a user
    user = User(**config_example.user_kwargs)
    # Define an assistant
    assistant = Leitner(**config_example.leitner_kwargs)
    # Bundle them together
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False,
                    reset_start_after=2, reset_go_to=3)
    bundle.reset()  # Reset in a state where the user has already produced an observation and made an inference.
    ## 0 : after assistant takes action + new task state
    ## 1 : after user observation + user inference + new user state
    ## 2 : after user takes action + new task state
    ## 3 : after assistant observation + assitant inference

    while True:
        state, rewards, is_done = bundle.step()
        if is_done:
            break

    print("Final reward", rewards['first_task_reward'])


if __name__ == "__main__":
    run_leitner()

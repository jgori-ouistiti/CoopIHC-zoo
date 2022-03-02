from coopihc import Bundle

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.leitner import Leitner
from coopihczoo.teaching.assistants.rl import Teacher


def run_leitner():
    n_item = 5
    inter_trial = 1
    max_iter = 40
    is_item_specific = False
    param = 0.01, 0.2

    delay_min = 1
    delay_factor = 2

    # Define a task
    task = Task(inter_trial=inter_trial, n_item=n_item, max_iter=max_iter)
    print("create task")
    print(task.state)
    # Define a user
    print("create user")
    user = User(n_item=n_item, is_item_specific=is_item_specific, param=param)
    print(user.state)
    # Define an assistant
    print("create assistant")
    assistant = Leitner(n_item=n_item, delay_factor=delay_factor, delay_min=delay_min)
    print(assistant.state)
    # Bundle them together
    print("create bundle")
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False)
    print(bundle.game_state)
    # Reset the bundle (i.e. initialize it to a random or prescribed states)
    print("reset bundle")

    ## 0 : after assistant takes action + new task state
    ## 1 : after user observation + user inference + new user state
    ## 2 : after user takes action + new task state
    ## 3 : after assistant observation + assitant inference
    bundle.reset(
        turn=3,
        skip_user_step=True
    )  # Reset in a state where the user has already produced an observation and made an inference.
    print(bundle.game_state)
    # Step through the bundle (i.e. play full rounds)
    print("start task")
    k = 0
    while 1:
        k += 1
        print(k)
        state, rewards, is_done = bundle.step(user_action=None,
                                              assistant_action=None)
                                              # go_to_turn=1)
        print(state)
        # Do something with the state or the rewards
        if is_done:
            break


def run_rl():

    n_item = 5
    inter_trial = 1
    max_iter = 40
    is_item_specific = False
    param = 0.01, 0.2

    thr = 0.9

    # Define a task
    task = Task(inter_trial=inter_trial, n_item=n_item, max_iter=max_iter)
    print("create task")
    print(task.state)
    # Define a user
    print("create user")
    user = User(n_item=n_item, is_item_specific=is_item_specific, param=param)
    print(user.state)
    # Define an assistant
    print("create assistant")
    assistant = Teacher(n_item=n_item, thr=thr)
    print(assistant.state)
    # Bundle them together
    print("create bundle")
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False)
    print(bundle.game_state)
    # Reset the bundle (i.e. initialize it to a random or prescribed states)
    print("reset bundle")

    ## 0 : after assistant takes action + new task state
    ## 1 : after user observation + user inference + new user state
    ## 2 : after user takes action + new task state
    ## 3 : after assistant observation + assitant inference
    bundle.reset(
        turn=3,
        skip_user_step=True
    )  # Reset in a state where the user has already produced an observation and made an inference.
    print(bundle.game_state)
    # Step through the bundle (i.e. play full rounds)
    print("start task")
    k = 0
    while 1:
        k += 1
        print(k)
        state, rewards, is_done = bundle.step(user_action=None,
                                              assistant_action=None)
                                              # go_to_turn=1)
        print(state)
        # Do something with the state or the rewards
        if is_done:
            break


if __name__ == "__main__":
    run_rl()

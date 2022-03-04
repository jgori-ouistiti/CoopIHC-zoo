from coopihc import Bundle

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.conservative_sampling import ConservativeSampling


def run_conservative():

    n_item = 5
    inter_trial = 1
    n_iter_per_ss = 40
    break_length = 1
    n_session = 1
    time_before_exam = 1
    is_item_specific = False
    param = 0.01, 0.2

    thr = 0.9  # Only for reward computation

    # Define a task
    task = Task(
        n_item=n_item,
        inter_trial=inter_trial,
        break_length=break_length,
        n_session=n_session,
        n_iter_per_ss=n_iter_per_ss,
        time_before_exam=time_before_exam,
        is_item_specific=is_item_specific,
        thr=thr)
    # Define a user
    user = User(param=param)
    # Define an assistant
    assistant = ConservativeSampling()
    # Bundle them together
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False)
    # Reset the bundle (i.e. initialize it to a random or prescribed states)
    ## 0 : after assistant takes action + new task state
    ## 1 : after user observation + user inference + new user state
    ## 2 : after user takes action + new task state
    ## 3 : after assistant observation + assitant inference
    bundle.reset(
        turn=3, skip_user_step=True
    )  # Reset in a state where the user has already produced an observation and made an inference.
    # Step through the bundle (i.e. play full rounds)
    k = 0
    while 1:
        k += 1
        print(k)
        state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
        # go_to_turn=1)
        print(state)
        # Do something with the state or the rewards
        if is_done:
            break


if __name__ == "__main__":
    run_conservative()

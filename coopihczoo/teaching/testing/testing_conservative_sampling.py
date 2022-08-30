from coopihc import Bundle

from coopihczoo.teaching.users.users_naive_implementation import User
from coopihczoo.teaching.envs.envs_naive_implementation import Task
from coopihczoo.teaching.assistants.assistants_naive_implementation.conservative_sampling import ConservativeSampling

from coopihczoo.teaching.config import config_example

task = Task(**config_example.task_kwargs)
user = User(**config_example.user_kwargs)
assistant = ConservativeSampling()

bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False,
                reset_start_after=2,
                reset_go_to=3)  # Begin by assistant

# Reset the bundle (i.e. initialize it to a random or prescribed states)

## 0 : after assistant takes action + new task state
## 1 : after user observation + user inference + new user state
## 2 : after user takes action + new task state
## 3 : after assistant observation + assitant inference

bundle.reset()
while True:
    # turn-index 3
    _state, _rewards, _is_done = bundle.quarter_step()
    print(_state.assistant_action)
    # turn-index 0
    _state, _rewards, _is_done = bundle.quarter_step()
    # turn-index 1
    _state, _rewards, _is_done = bundle.quarter_step()
    # turn-index 2
    state, rewards, is_done = bundle.quarter_step()   # Give user_action = X if you want to force the action

    # state, rewards, is_done = bundle.step()
    if is_done:
        break

# print("Final reward", rewards['first_task_reward'])
# print(bundle.state)
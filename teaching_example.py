from coopihc import Bundle

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants import Assistant

n_item = 5
is_item_specific = False
param = 0.01, 0.2

# Define a task
task = Task(n_item=n_item)
print("create task")
print(task.state)
# Define a user
print("create user")
user = User(n_item=n_item, is_item_specific=is_item_specific, param=param)
print(user.state)
# Define an assistant
print("create assistant")
assistant = Assistant(n_item=n_item)
print(assistant.state)
# Bundle them together
print("create bundle")
bundle = Bundle(task=task, user=user, assistant=assistant)
print(bundle.game_state)
# Reset the bundle (i.e. initialize it to a random or prescribed states)
print("reset bundle")
bundle.reset(
    turn=3
)  # Reset in a state where the user has already produced an observation and made an inference.

# Step through the bundle (i.e. play full rounds)
print("start task")
k = 0
while 1:
    k += 1
    print(k)
    state, rewards, is_done = bundle.step(user_action=1, assistant_action=None)
    print(state)
    # Do something with the state or the rewards
    if is_done:
        break

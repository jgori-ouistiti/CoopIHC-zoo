from coopihc import Bundle, ExampleTask, ExampleUser, ExampleAssistant

# Define a task, user, and assistant, and bundle them
example_task = ExampleTask()
example_user = ExampleUser()
example_assistant = ExampleAssistant()
bundle = Bundle(task=example_task, user=example_user, assistant=example_assistant)

# Initialize bundle (reset)
bundle.reset()

############# Main interaction loop (play rounds)
while True:
    # impose user action = 1, but sample assistant action from the assistant policy
    state, rewards, is_done = bundle.step(user_action=1, assistant_action=None)
    ########
    # Do something with the state or the rewards here
    ########
    if is_done:
        break

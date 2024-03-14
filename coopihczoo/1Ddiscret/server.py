from envs.envs import (
    DiscretePointingTaskPipeWrapper,
    SimplePointingTask,
)
from users.users import User
from assistants.assistants import ConstantCDGain

from coopihc import discrete_array_element, State
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.bundle.Bundle import Bundle
from coopihc.bundle.WsServer import WsServer


# Start task
task = SimplePointingTask(gridsize=20, number_of_targets=5)

# Define a user model defined elsewhere, but plug policy described just above inside to be used instead
user = User()
assistant = ConstantCDGain(1)
bundle = Bundle(task=task, user=user, assistant=assistant)
server = WsServer(
    bundle, DiscretePointingTaskPipeWrapper, address="localhost", port=8000
)
server.start()
# Now run launch coopihc/pointing/task_bundles.html in a browser and hit F5 to reset the task.

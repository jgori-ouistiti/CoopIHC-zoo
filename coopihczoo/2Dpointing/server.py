from envs.envs import (
    DiscretePointingTaskPipeWrapper,
    TwoDPointingTask,
)
from user.users import User
from assistants.assistants import ConstantCDGain, BIGGain

from coopihc import discrete_array_element, State
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.bundle.Bundle import Bundle
from coopihc.bundle.WsServer import WsServer


# Start task
task = TwoDPointingTask(gridsize=(10,10), number_of_targets=25)

# Define a user model defined elsewhere, but plug policy described just above inside to be used instead
# policy=BasePolicy(State(numpy.ARRAY([-1,1])))
# user = User(override_policy=policy)
user = User()
assistant = ConstantCDGain(1)
bundle = Bundle(task=task, user=user, assistant=assistant)
server = WsServer(
    bundle, DiscretePointingTaskPipeWrapper, address="localhost", port=4000
)
server.start()
# Now run launch task_bundles.html in a browser and hit F5 to reset the task.

from coopihc.bundle.Bundle import Bundle

from coopihczoo.pointing.envs.envs import SimplePointingTask
from coopihczoo.pointing.users.users import CarefulPointer
from coopihczoo.pointing.assistants.assistants import ConstantCDGain

task = SimplePointingTask(gridsize=31, number_of_targets=8)
binary_user = CarefulPointer(error_rate=0.4)
unitcdgain = ConstantCDGain(1)
bundle = Bundle(task=task, user=binary_user, assistant=unitcdgain)
bundle_state = bundle.reset(go_to=1)
bundle.render("plotext")
k = 0
while True:
    k += 1
    bundle_state, rewards, is_done = bundle.step(
        user_action=binary_user.take_action()[0], assistant_action=None
    )
    bundle.render("plotext")
    if is_done:
        bundle.close()
        break

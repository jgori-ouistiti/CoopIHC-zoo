from coopihczoo.pointing.envs import SimplePointingTask
from coopihczoo.pointing.users import CarefulPointer
from coopihczoo.pointing.assistants import ConstantCDGain
from coopihc.bundle.Bundle import Bundle


task = SimplePointingTask(gridsize=31, number_of_targets=8)
binary_user = CarefulPointer(error_rate=0.05)
unitcdgain = ConstantCDGain(1)
bundle = Bundle(task=task, user=binary_user, assistant=unitcdgain)
game_state = bundle.reset(turn=1)
bundle.render("plotext")
k = 0
while True:
    k += 1
    game_state, rewards, is_done = bundle.step(binary_user._take_action()[0], None)
    bundle.render("plotext")
    if is_done:
        bundle.close()
        break

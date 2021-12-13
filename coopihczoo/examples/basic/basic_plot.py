from coopihczoo.pointing.envs import SimplePointingTask
from coopihczoo.pointing.users import CarefulPointer
from coopihczoo.pointing.assistants import ConstantCDGain
from coopihczoo.bundle import Bundle


task = SimplePointingTask(gridsize=31, number_of_targets=8)
user = CarefulPointer()
assistant = ConstantCDGain(1)
bundle = Bundle(task=task, user=user, assistant=assistant)

game_state = bundle.reset()
bundle.render("plotext")


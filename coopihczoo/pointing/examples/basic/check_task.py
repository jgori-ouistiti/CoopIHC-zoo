from coopihczoo.pointing.envs import SimplePointingTask
from coopihc.bundle.Bundle import Bundle


task = SimplePointingTask(gridsize=31, number_of_targets=8, mode="position")
bundle = Bundle(task=task)
game_state = bundle.reset()
print(game_state)
# >>> print(game_state)
# ----------------  -----------  -------------------------  ------------------------------------------
# game_info         turn_index   0                          Discr(4)
#                   round_index  0                          Discr(2)
# task_state        position     7                          Discr(31)
#                   targets      [ 2  3  8 11 17 20 22 23]  MultiDiscr[31, 31, 31, 31, 31, 31, 31, 31]
# user_action       action       1                          Discr(2)
# assistant_action  action       1                          Discr(2)
# ----------------  -----------  -------------------------  ------------------------------------------
bundle.step(user_action=1, assistant_action=18)
print(bundle.game_state)
# ----------------  -----------  -------------------------  ------------------------------------------
# game_info         turn_index   0                          Discr(4)
#                   round_index  1                          Discr(2)
# task_state        position     18                         Discr(31)
#                   targets      [ 2  3  8 11 17 20 22 23]  MultiDiscr[31, 31, 31, 31, 31, 31, 31, 31]
# user_action       action       1                          Discr(2)
# assistant_action  action       18                         Discr(2)
# ----------------  -----------  -------------------------  ------------------------------------------

# Draft test in progress

from coopihc import Bundle, State, discrete_array_element, BaseAgent
from coopihczoo.pointing.envs.envs import SimplePointingTask


import random
import numpy
import copy

task = None


def test_init():
    global task
    gridnumber = random.randint(10, 100)
    nb_targets = random.randint(3, gridnumber)
    task = SimplePointingTask(
        gridsize=gridnumber, number_of_targets=nb_targets, mode="gain"
    )
    assert task.state["position"].space.low == 0
    assert task.state["position"].space.high == gridnumber - 1

    assert task.state["targets"].space.shape[0] == nb_targets


def test_reset():
    # lenient test
    global task
    task.reset()


def test_step():
    global task
    agent_state = State(goal=discrete_array_element(init=0))
    user = BaseAgent(agent_state=agent_state, role="user")
    bundle = Bundle(task=task, user=user, assistant=BaseAgent(role="assistant"))
    if task.mode == "gain":
        for i in range(100):
            bundle.reset(dic={"task_state": {"position": numpy.array([7])}})
            position = copy.copy(task.state["position"][0])
            user_action = random.randint(0, 10) - 5
            assistant_action = random.random() * 2
            game_state, reward, flag = bundle.step(
                user_action=user_action, assistant_action=assistant_action
            )

            new_pos = numpy.clip(
                numpy.round(
                    position + user_action * assistant_action,
                ),
                task.state["position"].space.low,
                task.state["position"].space.high,
            )

            assert game_state["task_state"]["position"] == new_pos


if __name__ == "__main__":
    test_init()
    test_reset()
    test_step()

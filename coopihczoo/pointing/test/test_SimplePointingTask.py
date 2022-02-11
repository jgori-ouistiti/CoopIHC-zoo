# Draft test in progress

from coopihc.bundle.Bundle import Bundle
from coopihczoo.pointing.envs import SimplePointingTask

import random
import numpy

task = None


def test_init():
    global task
    gridnumber = random.randint(10, 100)
    nb_targets = random.randint(2, gridnumber)
    task = SimplePointingTask(
        gridsize=gridnumber, number_of_targets=nb_targets, mode="gain"
    )
    assert task.state["position"].spaces.low == 0
    assert task.state["position"].spaces.high == gridnumber - 1

    assert len(task.state["targets"].spaces) == nb_targets


def test_reset():
    # lenient test
    global task
    task.reset()


def test_step():
    global task
    bundle = Bundle(task=task)
    if task.mode == "gain":
        for i in range(100):
            bundle.reset(dic={"task_state": {"position": numpy.array([0])}})
            position = task.state["position"][0]
            user_action = random.randint(0, 10) - 5
            assistant_action = random.random() * 2
            game_state, reward, flag = bundle.step(
                user_action=user_action, assistant_action=assistant_action
            )
            if game_state["task_state"]["position"] != numpy.round(
                user_action * assistant_action
            ):
                print(position, user_action, assistant_action)
                print(game_state["task_state"]["position"][0])
                print(numpy.round(user_action * assistant_action))
            assert game_state["task_state"]["position"] == numpy.round(
                user_action * assistant_action
            )


if __name__ == "__main__":
    test_init()
    test_reset()
    test_step()

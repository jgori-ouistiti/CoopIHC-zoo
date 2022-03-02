from coopihc import Bundle, TrainGym

from coopihczoo.teaching.users import User
from coopihczoo.teaching.envs import Task
from coopihczoo.teaching.assistants.rl import Teacher


def run_rl():

    n_item = 5
    inter_trial = 1
    max_iter = 40
    is_item_specific = False
    param = 0.01, 0.2
    thr = 0.9

    task = Task(
        inter_trial=inter_trial,
        n_item=n_item,
        max_iter=max_iter,
        is_item_specific=is_item_specific,
    )
    user = User(param=param)
    assistant = Teacher(thr=thr)
    bundle = Bundle(task=task, user=user, assistant=assistant, random_reset=False)

    env = TrainGym(
        bundle,
        train_user=False,
        train_assistant=True,
    )
    obs = env.reset()
    print(obs)
    print()
    print(env.action_space)
    # Dict(user_action:Discrete(3))
    print()
    print(env.observation_space)
    # Dict(turn_index:Discrete(4), round_index:Discrete(1000), position:Discrete(31), targets:MultiDiscrete([31 31 31 31 31 31 31 31]), goal:Discrete(31), user_action:Discrete(3), assistant_action:Box(1.0, 1.0, (1, 1), float32))
    env.step({"assistant_action": 1})

    # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    from stable_baselines3.common.env_checker import check_env

    check_env(env, warn=False)


if __name__ == "__main__":
    run_rl()

import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from coopihczoo.utils.imitation.dagger import DAgger
from coopihczoo.utils.imitation.behavioral_cloning import BC
from coopihczoo.utils.imitation.behavioral_cloning import sample_expert


def _get_expert(saving_path,
                make_env,
                expert_kwargs,
                expert_total_timesteps,
                eval_expert):

    env = make_env()

    if os.path.exists(f"{saving_path}.zip"):

        print("Loading the model...")
        expert = PPO.load(saving_path)

    else:
        expert = PPO(env=env, **expert_kwargs)
        reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
        print(f"Reward expert before training: {reward}")

        print("Training the expert...")
        expert.learn(total_timesteps=expert_total_timesteps)  # Note: set to 100000 to train a proficient expert

        print("Saving the expert...")
        os.makedirs("tmp", exist_ok=True)
        expert.save(saving_path)
        # expert.load(f"tmp/lunar_lander_expert_{int(expert_total_timesteps)}")

    if eval_expert:
        print("Evaluating expert...")
        reward, _ = evaluate_policy(expert.policy, Monitor(env), n_eval_episodes=50)
        print(f"Reward expert after training: {reward}")

    return expert


def train_novice_behavioral_cloning_ppo(
        make_env,
        expert_data,
        novice_kwargs):

    env = make_env()
    novice = PPO(env=env, **novice_kwargs)

    print("Evaluating the novice before training...")
    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice before training: {reward}")

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_data,
        policy=novice.policy)

    print("Training the novice's policy using behavior cloning...")
    bc_trainer.train()

    reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
    print(f"Reward novice after training: {reward}")

    return novice


def run_behavioral_cloning_ppo(
        saving_path,
        make_env,
        eval_expert=True,
        expert_total_timesteps=100000,
        expert_kwargs=None,
        novice_kwargs=None,
        sample_expert_n_episode=None,
        sample_expert_n_timestep=None
):
    if novice_kwargs is None:
        novice_kwargs = expert_kwargs

    expert = _get_expert(
        saving_path=saving_path,
        make_env=make_env,
        expert_kwargs=expert_kwargs,
        expert_total_timesteps=expert_total_timesteps,
        eval_expert=eval_expert)

    env = make_env()

    expert_data = sample_expert(env=env, expert=expert,
                                n_episode=sample_expert_n_episode,
                                n_timestep=sample_expert_n_timestep)

    novice = train_novice_behavioral_cloning_ppo(
        make_env=make_env,
        novice_kwargs=novice_kwargs,
        expert_data=expert_data)

    return novice


def train_novice_dagger_ppo(
        total_timesteps,
        make_env,
        novice_kwargs,
        expert,
        evaluate_novice=True,
        batch_size=32):

    env = make_env()
    novice = PPO(env=env, **novice_kwargs)

    if evaluate_novice:
        print("Evaluating the novice...")
        reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
        print(f"Reward novice before training: {reward}")

    dagger_trainer = DAgger(
        env=env,
        expert=expert,
        policy=novice.policy,
        batch_size=batch_size)

    print("Training the novice's policy using dagger...")
    dagger_trainer.train(total_timesteps=total_timesteps)

    if evaluate_novice:
        print("Evaluating the novice...")
        reward, _ = evaluate_policy(novice.policy, Monitor(env), n_eval_episodes=50)
        print(f"Reward novice after training: {reward}")

    return novice


def run_dagger_ppo(
        saving_path,
        make_env,
        eval_expert=True,
        expert_total_timesteps=100000,
        dagger_total_timesteps=5000,
        expert_kwargs=None,
        novice_kwargs=None):

    if novice_kwargs is None:
        novice_kwargs = expert_kwargs

    expert = _get_expert(
        saving_path=saving_path,
        make_env=make_env,
        expert_kwargs=expert_kwargs,
        expert_total_timesteps=expert_total_timesteps,
        eval_expert=eval_expert)

    novice = train_novice_dagger_ppo(
        total_timesteps=dagger_total_timesteps,
        expert=expert,
        novice_kwargs=novice_kwargs,
        make_env=make_env)

    return novice

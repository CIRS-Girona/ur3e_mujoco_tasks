import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,TD3
import os
from pathlib import Path
import argparse

register(
    id="ur3e_tasks/UR3ePegInHoleEnv-v0",
    entry_point="ur3e_tasks.envs:UR3ePegInHoleEnv",
    max_episode_steps=250, # maximum number of steps per episode
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to load a trained DRL model for peg-in-hole task.")

    # Define arguments
    parser.add_argument('--algorithm', type=str, required=True, help='The algorithm used (SAC, TD3)')
    parser.add_argument('--filename', type=str, required=True, help='The name of the file to load (NOTE: algorithm must match --algorithm)')
    parser.add_argument('--learning-stage', type=int, required=False, default=1, help='Learning stage to test (default=1)')

    parser.add_argument('--num-episodes', type=int, required=False, help='Number of episodes to test (infinite if not specified)')

    # Parse arguments
    args = parser.parse_args()

    return args.algorithm, args.filename, args.learning_stage, args.num_episodes

def main():
    algorithm, filename, stage, num_episodes = parse_arguments()
    # Create the environment with rendering in human mode
    env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode='human')

    # check_env(env)

    FILE_DIR = Path(__file__).parent /'..' / 'models'
    # Make sure the directory exists
    os.makedirs(FILE_DIR, exist_ok=True)
    file_path = os.path.join(FILE_DIR,filename)

    # instantiate and train model
    if algorithm == 'SAC':
        alg_func = SAC
    elif algorithm == 'TD3':
        alg_func = TD3
    else:
        raise "Algorithm is not valid!"

    ## Testing

    try:
        print(f"Loading model from {os.path.abspath(file_path)} ...")
        model = alg_func.load(file_path)
    except:
        raise "Error loading model!"
    
    # set environment learning stage
    env.unwrapped.set_learning_stage(stage)

    obs, info = env.reset()

    # initialize stats for logging
    i = 0
    total_reward = 0
    total_ep_reward = 0
    success_count = 0
    total_ep = 0
    total_ep_len = 0

    if num_episodes is None:
        num_episodes = float("inf")  # infinite episodes if not specified

    # variable to store max contact force
    max_contact_force = np.zeros(6)

    while True:
        i += 1
        # generate action from policy
        action, _ = model.predict(obs, deterministic=True)
        # print(f"action = {action}")
        # Take a step in the environment using the chosen action
        obs, reward, terminated, truncated, info = env.step(action)
        # Store episode reward
        total_ep_reward += reward

        # track max contact force
        for j in range(6):
            if abs(obs[j]) > abs(max_contact_force[j]):
                max_contact_force[j] = obs[j]

        # Check if the episode is over (terminated)
        if terminated or truncated:
            # logging
            total_ep += 1
            total_reward += total_ep_reward
            success_count += 1 if info["is_success"] else 0
            total_ep_len += i
            print("==================")
            print(f"Total episode reward = {total_ep_reward}")
            print(f"Average episode reward after {total_ep} episodes = {total_reward/total_ep}")
            print(f"Success rate = {success_count/total_ep}")
            print(f"Average episode length = {total_ep_len/total_ep}")
            print(f"Max contact force = {max_contact_force}")

            if total_ep >= num_episodes:
                print("Finished testing!")
                env.close()
                break
            
            # reset environment and episode stats
            obs, info = env.reset()
            i = 0
            total_ep_reward = 0
            max_contact_force = np.zeros(6)
        # print("==================")

if __name__ == "__main__":
    main()
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
    # Optionally, you can set a maximum number of steps per episode
    max_episode_steps=1000,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to load a trained DRL model for peg-in-hole task.")

    # Define arguments
    parser.add_argument('--algorithm', type=str, required=True, help='The algorithm used (SAC, TD3)')
    parser.add_argument('--filename', type=str, required=True, help='The name of the file to load (NOTE: algorithm must match --algorithm)')

    # Parse arguments
    args = parser.parse_args()

    return args.algorithm, args.filename

def main():
    algorithm, filename = parse_arguments()
    # Create the environment with rendering in human mode
    env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode='human')

    # check_env(env)

    FILE_DIR = Path(__file__).parent /'..' / 'models'
    # Make sure the directory exists
    os.makedirs(FILE_DIR, exist_ok=True)

    # instantiate and train model
    if algorithm == 'SAC':
        alg_func = SAC
    elif algorithm == 'TD3':
        alg_func = TD3
    else:
        raise "Algorithm is not valid!"

    ## Testing

    try:
        model = alg_func.load(os.path.join(FILE_DIR,filename))
    except:
        raise "Error loading model!"

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
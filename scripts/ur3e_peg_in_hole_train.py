import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3
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
    parser = argparse.ArgumentParser(description="Script to train a DRL model for peg-in-hole task.")

    # Define arguments
    parser.add_argument('--algorithm', type=str, required=True, help='The algorithm to use (SAC, TD3)')
    parser.add_argument('--save-filename', type=str, required=True, help='The name of the file to save the model to')
    parser.add_argument('--enable-log', action='store_true', help='Enable logging.')

    # Parse arguments
    args = parser.parse_args()

    return args.algorithm, args.save_filename, args.enable_log

def main():
    algorithm, filename, logging = parse_arguments()
    # Create the environment with rendering in human mode
    env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode='human')

    # check_env(env)

    ## Training Phase

    # directory and file name to save trained model
    SAVE_DIR = Path(__file__).parent /'..' / 'models'
    # Make sure the directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    if logging:
        # directory for logging
        LOG_DIR = Path(__file__).parent /'..' / 'log'
        os.makedirs(LOG_DIR, exist_ok=True)
    else:
        LOG_DIR = None

    # instantiate and train model
    if algorithm == 'SAC':
        alg_func = SAC
    elif algorithm == 'TD3':
        alg_func = TD3
    else:
        raise "Algorithm is not valid!"

    model = alg_func("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=5000, log_interval=4, tb_log_name=filename)
    model.save(os.path.join(SAVE_DIR,filename))

if __name__ == "__main__":
    main()
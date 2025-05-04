import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3
import os
from pathlib import Path
import argparse

# from ur3e_tasks.utils.callbacks import SuccessTrackerCallback

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
    parser.add_argument('--total-timesteps', type=int, required=False, default=10000, help='Total timesteps to train')
    parser.add_argument('--render', action='store_true', help='Enable renderring.')
    parser.add_argument('--enable-log', action='store_true', help='Enable logging.')

    # Parse arguments
    args = parser.parse_args()

    # return args.algorithm, args.save_filename, args.render, args.enable_log
    return args

def main():
    # algorithm, filename, render, logging = parse_arguments()
    args = parse_arguments()

    render_mode = 'human' if args.render else None

    # Create the environment with rendering in human mode
    env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode=render_mode)

    # check_env(env)

    ## Training Phase

    # directory and file name to save trained model
    SAVE_DIR = Path(__file__).parent /'..' / 'models'
    # Make sure the directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR,args.save_filename)

    if args.enable_log:
        # directory for logging
        LOG_DIR = Path(__file__).parent /'..' / 'log'
        os.makedirs(LOG_DIR, exist_ok=True)
    else:
        LOG_DIR = None

    # instantiate and train model
    if args.algorithm == 'SAC':
        alg_func = SAC
    elif args.algorithm == 'TD3':
        alg_func = TD3
    else:
        raise "Algorithm is not valid!"

    model = alg_func("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=args.total_timesteps, log_interval=4, tb_log_name=args.save_filename)
    model.save(save_path)
    print(f"Model saved in {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()
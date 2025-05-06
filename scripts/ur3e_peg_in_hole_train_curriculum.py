import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3
import os
from pathlib import Path
import argparse

from ur3e_tasks.utils.callbacks import SuccessTrackerCallback

register(
    id="ur3e_tasks/UR3ePegInHoleEnv-v0",
    entry_point="ur3e_tasks.envs:UR3ePegInHoleEnv",
    # Optionally, you can set a maximum number of steps per episode
    max_episode_steps=250,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train a DRL model with curriculum learning for peg-in-hole task.")

    # Define arguments
    parser.add_argument('--algorithm', type=str, required=True, help='The algorithm to use (SAC, TD3)')
    parser.add_argument('--filename', type=str, required=True, help='The name of the folder to save the model to, or if --continue-training is enabled, the name of folder to load model.')
    parser.add_argument('--total-timesteps', type=int, required=False, default=10000, help='Total timesteps to train')
    parser.add_argument('--render', action='store_true', help='Enable renderring.')

    parser.add_argument('--continue-training', action='store_true', help='Continue training from a saved model')
    parser.add_argument('--starting-stage', type=int, required=False, default=1, help='Curriculum learning stage to start training with')
    # parser.add_argument('--enable-log', action='store_true', help='Enable logging.')

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

    # set environment learning stage
    env.unwrapped.set_learning_stage(args.starting_stage)

    # check_env(env)

    ## Training Phase
    # directory and file name to save trained model
    SAVE_DIR = Path(__file__).parent /'..' / 'models'
    # Make sure the directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR,args.filename)

    # directory for logging
    LOG_DIR = Path(__file__).parent /'..' / 'log'
    log_path = os.path.join(LOG_DIR,args.filename)
    os.makedirs(log_path, exist_ok=True)


    # instantiate and train model
    if args.algorithm == 'SAC':
        alg_func = SAC
    elif args.algorithm == 'TD3':
        alg_func = TD3
    else:
        raise "Algorithm is not valid!"


    # load model if continue training is requested
    if args.continue_training:
        # search for the last model
        for file in os.listdir(save_path):
            full_path = os.path.join(save_path, file)
            if os.path.isfile(full_path) and 'final' in file:
                model_loadpath = full_path
        # load that model
        model = alg_func.load(model_loadpath, tensorboard_log=log_path)
        model.verbose = 1
        model.set_env(env)
    else:
        model = alg_func("MlpPolicy", env, verbose=1, tensorboard_log=log_path)


    callback = SuccessTrackerCallback(
        n_episodes=50,
        next_stage_threshold=90,
        log_dir=log_path,
        save_path=save_path,
        verbose=1
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name="general", reset_num_timesteps=False)

if __name__ == "__main__":
    main()
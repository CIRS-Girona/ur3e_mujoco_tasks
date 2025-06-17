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
    parser.add_argument('--seed', type=int, required=False, help='Random seed for reproducibility')

    # Parse arguments
    args = parser.parse_args()

    # return args.algorithm, args.filename, args.learning_stage, args.num_episodes
    return args

def main():
    # algorithm, filename, stage, num_episodes = parse_arguments()
    args = parse_arguments()

    # flags for saving data (NOTE: toggle on/off as needed)
    save_trajectory = False
    save_force = False
    save_action = False

    # Create the environment with rendering in human mode
    env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode='human')

    # check_env(env)

    FILE_DIR = Path(__file__).parent /'..' / 'models'
    # Make sure the directory exists
    os.makedirs(FILE_DIR, exist_ok=True)
    file_path = os.path.join(FILE_DIR,args.filename)

    # instantiate and train model
    if args.algorithm == 'SAC':
        alg_func = SAC
    elif args.algorithm == 'TD3':
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
    env.unwrapped.set_learning_stage(args.learning_stage)

    obs, info = env.reset(seed=args.seed)

    # initialize stats for logging
    i = 0
    total_reward = 0
    total_ep_reward = 0
    success_count = 0
    total_ep = 0
    total_ep_len = 0

    num_episodes = args.num_episodes
    if num_episodes is None:
        num_episodes = float("inf")  # infinite episodes if not specified

    # variable to store max contact force
    max_contact_force = np.zeros(6)
    sum_max_contact_force = np.zeros(6)

    # store peg and hole position for plotting
    hole_pos = info["hole_pos_real"]
    peg_pos = obs[6:9]

    # store force for plotting
    ee_force = obs[:3]

    # store action for plotting
    action_evol = None

    print(f"Hole pos = {hole_pos}")

    while True:
        i += 1
        # generate action from policy
        action, _ = model.predict(obs, deterministic=True)
        if action_evol is None:
            action_evol = action
        else:
            action_evol = np.vstack((action_evol,action))
        # print(f"action = {action}")
        # Take a step in the environment using the chosen action
        obs, reward, terminated, truncated, info = env.step(action)
        # Store episode reward
        total_ep_reward += reward

        # track max contact force
        for j in range(6):
            if abs(obs[j]) > abs(max_contact_force[j]):
                max_contact_force[j] = abs(obs[j])

        # store trajectory for plotting
        peg_pos = np.vstack((peg_pos,obs[6:9]))
        # store ee force for plotting
        ee_force = np.vstack((ee_force,obs[:3]))

        # Check if the episode is over (terminated)
        if terminated or truncated:
            # logging
            total_ep += 1
            total_reward += total_ep_reward
            success_count += 1 if info["is_success"] else 0
            total_ep_len += i
            sum_max_contact_force += max_contact_force

            print("==================")
            print(f"Total episode reward = {total_ep_reward}")
            print(f"Max contact force = {max_contact_force}")
            print(f"Stats after {total_ep} episodes:")
            print(f"Average episode reward = {total_reward/total_ep}")
            print(f"Success rate = {success_count/total_ep}")
            print(f"Average episode length = {total_ep_len/total_ep}")
            print(f"Average maximum contact force = {sum_max_contact_force / total_ep}")
            
            if total_ep >= num_episodes:
                print("Finished testing!")
                env.close()

                # store trajectory in a csv file
                if save_trajectory:
                    SAVE_DIR = Path(__file__).parent /'..' / 'log' / 'test_traj'
                    # Make sure the directory exists
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    hole_filename = os.path.join(SAVE_DIR,"hole_pos_real.csv")
                    np.savetxt(hole_filename, hole_pos, delimiter=",")
                    traj_filename = os.path.join(SAVE_DIR,"peg_trajectory_real.csv")
                    np.savetxt(traj_filename, peg_pos, delimiter=",")

                    print("Hole position file saved in: ", os.path.abspath(hole_filename))
                    print("Trajectory file saved in: ", os.path.abspath(traj_filename))

                # store ee force in a csv file
                if save_force:
                    SAVE_DIR = Path(__file__).parent /'..' / 'log' / 'test_force'
                    # Make sure the directory exists
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    ee_force_filename = os.path.join(SAVE_DIR,"ee_force_real.csv")
                    np.savetxt(ee_force_filename, ee_force, delimiter=",")
                    print("End-effector force file saved in: ", os.path.abspath(ee_force_filename))

                # store actions in a csv file
                if save_action:
                    SAVE_DIR = Path(__file__).parent /'..' / 'log' / 'actions'
                    # Make sure the directory exists
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    actions_filename = os.path.join(SAVE_DIR,"actions_real.csv")
                    np.savetxt(actions_filename, action_evol, delimiter=",")
                    print("Actions file saved in: ", os.path.abspath(actions_filename))

                break
            
            # reset environment and episode stats
            obs, info = env.reset()
            i = 0
            total_ep_reward = 0
            max_contact_force = np.zeros(6)
        # print("==================")

if __name__ == "__main__":
    main()
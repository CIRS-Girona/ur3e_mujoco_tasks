import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import datetime
import pickle
import zipfile
import time

def collect(data_list, episode, output_dir="/home/tanakrit-ubuntu/ur3e_mujoco_tasks/scripts/data", timestamp=None, metadata=None):
    """
    Saves images and metadata from the observation, action, and info, then compresses the files.

    Args:
        data_list: List of dictionaries, each containing collected data for a step.
        episode: Episode number.
        output_dir: Directory where the data is saved.
        timestamp: Unique identifier for the data collection session.
    """
    zip_filename = f"{timestamp}_{episode}.zip"
    zip_path = os.path.join(output_dir, zip_filename)

    # Open the zip file in write mode
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, data in enumerate(data_list):
            # Create filename for each pickle file
            pkl_filename = f"{timestamp}_{episode}_{i}.pkl"
            pkl_path = os.path.join(output_dir, pkl_filename)

            # Save dictionary as pickle
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)

            print(f"Saved: {pkl_path}")

            # Add file to zip archive
            zipf.write(pkl_path, pkl_filename)

            # Remove the original pickle file after zipping (optional)
            os.remove(pkl_path)
    
    if metadata:
        pkl_filename = f"metadata_{timestamp}_{episode}.pkl"
        pkl_path = os.path.join(output_dir, pkl_filename)
        with open(pkl_path, "wb") as f:
            pickle.dump(metadata, f)
    

    print(f"All files zipped into: {zip_path}")
    return zip_path
    



def pack(obs, info,action,state):
    merged_dict = {**obs, **info}
    merged_dict["ee_vel"] = np.array(action).reshape(6,1)
    merged_dict["state"] = state

    print(merged_dict)
    return merged_dict

register(
    id="ur3e_tasks/UR3eAssemblyEnv-v0",
    entry_point="ur3e_tasks.envs:UR3eAssemblyEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
# Create the environment with rendering in human mode
env = gymnasium.make('ur3e_tasks/UR3eAssemblyEnv-v0', render_mode='human')
episode = 0
data_list = []
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# recored param
collect_period = 100
collect_step = -1
start_collect = False

while True:
    
    action,state,_, terminated = env.unwrapped.get_action_bt()
    # collect data

    # start collecting when robot is moving # prevent 0 in dataset
    if action is not None and sum(action) != 0:
        start_collect = True
        collect_step = collect_step +1

        
    if collect_step%collect_period == 0 and start_collect:
        data_dict = pack(observation, info,action,state )
        data_list.append(data_dict)
        

    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    
    # print(action)
    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # save trajectories if success
        if info["success"] ==1:
            print("Episode successful! --> save trajectories")
            collect(data_list,episode,timestamp=timestamp,metadata=env.unwrapped._random_state)
        else:
            print("Episode fail! --> discard trajectories")

        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()
        episode = episode +1
        data_list = []
        start_collect = False
        collect_step = -1

# Close the environment when the simulation is done
env.close()



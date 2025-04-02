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

def collect(data,episode, output_dir= "/home/tanakrit-ubuntu/ur3e_mujoco_tasks/scripts/data"):
    """
    Saves images and metadata from the observation, action, and info.
    
    Args:
        obs: List containing [img1 (np.array), img2 (np.array), img3 (np.array), ee_pose (np.array), jointstate (np.array)]
        action: List containing [vel_cmd (np.array)]
        info: Dictionary containing [hole_position (np.array), success (Boolean), state (String), trajectories (int), step (int)]
        execution_name: String representing the current execution name.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    step = data["step"]
    filename = f"{timestamp}_{episode}_{step}.pkl"

    # Full file path
    file_path = f"{output_dir}/{filename}"

    # Save dictionary as pickle
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved: {file_path}")

    return file_path
    



def pack(obs, info,action,state, success, terminated):
    merged_dict = {**obs, **info}
    merged_dict["ee_vel"] = np.array(action).reshape(6,1)
    merged_dict["state"] = state
    merged_dict["success"] = int(success)
    merged_dict["terminated"] = int(terminated)

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
# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

while True:
    
    action,state,success, terminated = env.unwrapped.get_action_bt()
    # collect data
    if info["step"]%100 == 0:
        data_dict = pack(observation, info,action,state,success, terminated )
        collect(data_dict,episode)

    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    
    # print(action)
    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()
        episode = episode +1

# Close the environment when the simulation is done
env.close()



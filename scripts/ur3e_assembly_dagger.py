import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import torch
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import datetime
import pickle
import zipfile
import time
from ur3e_bc.modules import UR3EDataset
from ur3e_bc.models import UR3EBCModel, UR3EBCRNNModel, UR3EBC3DModel
import multiprocessing as mp

def run_collect(process_id,model_path, output_dir):
    def collect(data_list, episode, output_dir="/home/students/ur3e_mujoco_tasks/data/data_near", timestamp=None, metadata=None):
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


    def load_model(model, path):
        # Load the checkpoint
        checkpoint = torch.load(path)
        
        # Load the model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Switch to evaluation mode
        model.eval()
        
        return model

    def action_error(expert, action):
        return np.linalg.norm(expert -action)


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
    timestamp = datetime.datetime.now().strftime(f"%Y%m%d_%H%M%S_proc{process_id}")

    ########################################################
    ### Model
    ########################################################
    # model = UR3EBCModel()
    # model = UR3EBCRNNModel(False)
    model = UR3EBC3DModel("mc")
    model_path = model_path
    # model_path = "/home/students/ur3e_behavior_cloning/runs/model_20250409_224215.pth"
    model = load_model(model,model_path)
    env.unwrapped._model =  model



    ########################################################
    ### Replay
    ########################################################
    # zip_dir = "/home/students/ur3e_mujoco_tasks/data/data_new"
    # dataset = UR3EDataset(zip_dir)
    # replay_name = "20250410_164438_449.zip" #446
    # replay = dataset._get_episode(replay_name)
    # metadata = dataset.metadata_dict[replay_name]
    # env.unwrapped.random_domain = False
    # # env.unwrapped.set_replay(replay,metadata) 

    # Reset the environment with a specific seed for reproducibility
    observation, info = env.reset(seed=42 + process_id)

    ########################################################
    ### Collect
    ########################################################
    output_dir=output_dir
    collect_period = 100
    collect_step = -1
    start_collect = False
    max_episodes = 500

    ########################################################
    ### Dagger
    ########################################################
    action_err_thresh = 0.05
    max_data = 100



    while True:
        action,hole,state = env.unwrapped.get_action_model(observation)
        action_exp,state_exp,_, terminated_exp = env.unwrapped.get_action_bt() # expert
        
        # start collecting when robot is moving # prevent 0 in dataset
        if action is not None and sum(action) != 0:
            start_collect = True
            collect_step = collect_step +1

            
        if collect_step%collect_period == 0 and start_collect:
            data_dict = pack(observation, info,action_exp,state_exp )
            data_list.append(data_dict)
            

        # Take a step in the environment using the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

        
        # print(action)
        # Check if the episode is over (terminated) or max steps reached (truncated)
        if terminated or truncated or  len(data_list) > max_data:
            # save trajectories
            collect(data_list,episode,output_dir=output_dir,timestamp=timestamp,metadata=env.unwrapped._random_state)

            # If the episode ends or is truncated, reset the environment
            observation, info = env.reset()
            episode = episode +1
            data_list = []
            start_collect = False
            collect_step = -1

            if episode > max_episodes:
                env.close()
                exit()



if __name__ == "__main__":
    NUM_PROCESSES = 4
    model_path = "/home/students/ur3e_behavioural_cloning/runs/3d_mc_4000data_20250414_235859.pth"
    output_dir = "/home/students/ur3e_mujoco_tasks/data/data_dagger"

    processes = []
    for i in range(NUM_PROCESSES):
        p = mp.Process(target=run_collect, args=(i, model_path, output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
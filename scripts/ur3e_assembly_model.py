import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import pickle
from ur3e_bc.modules import UR3EDataset
from ur3e_bc.models import UR3EBCModel, UR3EBCRNNModel, UR3EBC3DModel, UR3EFuseEarlyModel, UR3EBC3DModelModified

import torch

def load_model(model, path):
    # Load the checkpoint
    checkpoint = torch.load(path,weights_only=False)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Switch to evaluation mode
    model.eval()
    
    return model

register(
    id="ur3e_tasks/UR3eAssemblyEnv-v0",
    entry_point="ur3e_tasks.envs:UR3eAssemblyEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
# Create the environment with rendering in human mode
env = gymnasium.make('ur3e_tasks/UR3eAssemblyEnv-v0', render_mode='human')

# Model
# model = UR3EBCModel()
# model = UR3EBCRNNModel(True)
# model = UR3EBC3DModel("mc")
model = UR3EBC3DModelModified("mc")
# model = UR3EFuseEarlyModel("mc")
model = model.to("cuda")
model_path = "/home/students/ur3e_behavioural_cloning/runs/dagger/3d_mc_pose_separate_realign_continue_dagger_3_20250503_030531.pth"
# model_path = "/home/students/ur3e_behavior_cloning/runs/model_20250409_224215.pth"
model = load_model(model,model_path)

# set up env to replay
env.unwrapped._model =  model

zip_dir = "/home/students/ur3e_mujoco_tasks/data/data_new"
dataset = UR3EDataset(zip_dir)

# set up env to replay
replay_name = "20250410_164438_449.zip" #446
replay = dataset._get_episode(replay_name)
metadata = dataset.metadata_dict[replay_name]
env.unwrapped.random_domain = True
env.unwrapped.show_cam = False
# env.unwrapped.set_replay(replay,metadata) 

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# Run simulation for a fixed number of steps
# for _ in range(1000):
while True:
    # Choose a random action from the available action space
    # action = env.action_space.sample()
    action,hole,state = env.unwrapped.get_action_model(observation)
    # action_exp,state_exp,_, terminated_exp = env.unwrapped.get_action_bt() # expert
    action = action 

    # print("Action expert : {}".format(action_exp))
    # print("Action : {}".format(action))
    # print("Hole : {}".format(info["hole_pose"].reshape(7)))
    # print("Predicted Hole : {}".format(hole))
    # print("Predicted State : {}".format(state))
    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)
    # print(action)
    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()

# Close the environment when the simulation is done
env.close()

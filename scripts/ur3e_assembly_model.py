import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import pickle
from ur3e_bc.modules import UR3EDataset
from ur3e_bc.models import UR3EBCModel

import torch

def load_model(model, path):
    # Load the checkpoint
    checkpoint = torch.load(path,map_location=torch.device('cpu'))
    
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
model = UR3EBCModel()
model_path = "/home/tanakrit-ubuntu/ur3e_mujoco_tasks/scripts/model/model_20250409_160830.pth"
model = load_model(model,model_path)

# set up env to replay
env.unwrapped._model =  model



# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# Run simulation for a fixed number of steps
# for _ in range(1000):
while True:
    # Choose a random action from the available action space
    # action = env.action_space.sample()
    action,_,_ = env.unwrapped.get_action_model(observation)
    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)
    # print(action)
    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()

# Close the environment when the simulation is done
env.close()

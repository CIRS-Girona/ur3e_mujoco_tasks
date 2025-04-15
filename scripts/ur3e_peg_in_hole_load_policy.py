import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,TD3
import os
from pathlib import Path

register(
    id="ur3e_tasks/UR3ePegInHoleEnv-v0",
    entry_point="ur3e_tasks.envs:UR3ePegInHoleEnv",
    # Optionally, you can set a maximum number of steps per episode
    max_episode_steps=1000,
)
# Create the environment with rendering in human mode
env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode='human')

# check_env(env)

FILE_DIR = Path(__file__).parent /'..' / 'models'
# Make sure the directory exists
os.makedirs(FILE_DIR, exist_ok=True)

filename = os.path.join(FILE_DIR, 'peg_in_hole_td3_try1')

## Testing

model = TD3.load(filename)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
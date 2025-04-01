import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import numpy as np

register(
    id="ur3e_tasks/UR3ePegInHoleEnv-v0",
    entry_point="ur3e_tasks.envs:UR3ePegInHoleEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
# Create the environment with rendering in human mode
env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode='human')

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# Run simulation for a fixed number of steps
# for _ in range(1000):
i = 0
while True:
    i+=1
    # Choose a random action from the available action space
    action = env.action_space.sample()
    # action = np.array([0.5,0.5,0.1,0.0,0.0])
    print("action = ", action)
    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)
    # print("observation = ", observation)
    print("info =")
    print(info)
    print("reward = ", reward)
    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()
        i = 0

# Close the environment when the simulation is done
env.close()

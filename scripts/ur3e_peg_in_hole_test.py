import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import numpy as np

from manipulator_mujoco.utils.transform_utils import mat2quat, quat2axisangle, quat2mat

register(
    id="ur3e_tasks/UR3ePegInHoleEnv-v0",
    entry_point="ur3e_tasks.envs:UR3ePegInHoleEnv",
    # Optionally, you can set a maximum number of steps per episode
    max_episode_steps=500,
)

def invert_rotation(rot_matrix):
    # rotate hole frame (x,180)*(z,90)
    rotx180 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    rotz90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    return rot_matrix @ rotx180 @ rotz90

def generate_target_vel(obs,target):
    peg_end_pos = obs[6:9]
    peg_end_quat = obs[9:13] #wxyz
    peg_end_rot = quat2mat([peg_end_quat[1], peg_end_quat[2], peg_end_quat[3], peg_end_quat[0]])
    hole_pos = obs[13:16]
    hole_quat = obs[16:20] #wxyz
    hole_rot = quat2mat([hole_quat[1], hole_quat[2], hole_quat[3], hole_quat[0]])

    # set intermediate target position (above the hole)
    offset = hole_rot @ np.array([0,0,0.07]).T
    intermediate_pt = hole_pos + offset

    # align the frames bcs hole is z+ up, and peg is z+ down
    hole_rot_inverted = invert_rotation(hole_rot)

    # reproduce transformation matrix of peg (w.r.t. world)
    peg_end_transform = np.block([[peg_end_rot,peg_end_pos.reshape(-1,1)],[0,0,0,1]]) # transformation matrix

    # compute position and orientation of the intermediate target pos w.r.t. peg
    intermediate_pt_transform = np.block([[hole_rot_inverted,intermediate_pt.reshape(-1,1)],[0,0,0,1]])
    intermediate_pt_wrt_peg_transform = np.linalg.inv(peg_end_transform) @ intermediate_pt_transform
    intermediate_pt_wrt_peg_pos = intermediate_pt_wrt_peg_transform[:3,3]# extract position
    intermediate_pt_wrt_peg_rot = intermediate_pt_wrt_peg_transform[:3,:3] # rotation matrix
    intermediate_pt_wrt_peg_quat = mat2quat(intermediate_pt_wrt_peg_rot) # format: xyzw

    # express orientation error as norm of axis angle
    ori_error = quat2axisangle(intermediate_pt_wrt_peg_quat)
    ori_error_norm = np.linalg.norm(ori_error[:2]) # we don't care about the z axis

    # compute distance to intermediate target point
    distance_to_intermediate_pt = np.linalg.norm(intermediate_pt_wrt_peg_pos)

    # print(f"distance_to_intermediate_pt = {distance_to_intermediate_pt}")
    # print(f"ori_error_norm = {ori_error_norm}")

    # assign target and position error based on simulation state
    if (distance_to_intermediate_pt < 0.005 and ori_error_norm < 0.05) or target=="hole": # if peg end is already at the intermediate point and orientation aligns
        # compute pos error to hole instead
        hole_transform = np.block([[hole_rot,hole_pos.reshape(-1,1)],[0,0,0,1]]) # transformation matrix
        hole_wrt_peg_transform = np.linalg.inv(peg_end_transform) @ hole_transform
        pos_error = hole_wrt_peg_transform[:3,3]
        target = "hole"
    else:
        target = "intermediate"
        pos_error = intermediate_pt_wrt_peg_pos # take intermediate point

    control_error = np.concatenate([pos_error,ori_error])
    target_vel = 1.2 * control_error

    # print(f"target = {target}")

    # clip target vel
    limits = np.array([0.2,0.2,0.2,0.1,0.1])
    return np.clip(target_vel[:5],limits*-1,limits), target

# Create the environment with rendering in human mode
env = gymnasium.make('ur3e_tasks/UR3ePegInHoleEnv-v0', render_mode='human')
stage = 1
env.unwrapped.set_learning_stage(stage)

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# Run simulation for a fixed number of steps
# for _ in range(1000):
i = 0
total_reward = 0
total_ep_reward = 0
success_count = 0
total_ep = 0
total_ep_len = 0

target = "intermediate" # flag

while True:
    i+=1
    # Choose a random action from the available action space
    # action = env.action_space.sample()
    action,target = generate_target_vel(observation,target)
    # action = np.array([-0.3,0.3,-0.5,0.0,0.0])
    # print("action = ", action)
    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)
    total_ep_reward += reward
    # print("observation = ", observation)
    # print("reward = ", reward)
    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # logging
        total_ep += 1
        total_reward += total_ep_reward
        success_count += 1 if info["is_success"] else 0
        total_ep_len += i
        print("==================")
        print(f"Total episode reward = {total_ep_reward}")
        print(f"Average episode reward after {total_ep} episodes = {total_reward/total_ep}")
        print(f"Success rate = {success_count/total_ep}")
        print(f"Average episode length = {total_ep_len/total_ep}")
        # If the episode ends or is truncated, reset the environment
        # advance learning stage to debug all stages
        if total_ep % 5 == 0:
            stage += 1 if stage < 6 else 0
            env.unwrapped.set_learning_stage(stage)

        observation, info = env.reset()
        i = 0
        total_ep_reward = 0
        target = "intermediate"
    print("==================")

# Close the environment when the simulation is done
env.close()

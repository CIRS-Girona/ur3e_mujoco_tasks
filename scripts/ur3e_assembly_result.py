import gymnasium
import manipulator_mujoco
from gymnasium.envs.registration import register
import pickle
from ur3e_bc.modules import UR3EDataset
from ur3e_bc.models import UR3EBCModel, UR3EBCRNNModel, UR3EBC3DModel, UR3EFuseEarlyModel, UR3EBC3DModelModified
import math
import torch
import numpy as np

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


# Reuslt related
results_dict = {"states":[],
                "vel_error":[],
                "hole_error":[],
                "state_acc":[],
                "time1":[],
                "time2":[],
                "time3":[],
                "time4":[],}

hole_err = []
vel_err = []
state_acc = []
time = [0,0,0,0]


# Reusult param
test_eps = 2
eps_counter = 0
step_limit = 7000 
step_counter = 0
previous_state = 0

# Create the environment with rendering in human mode
env = gymnasium.make('ur3e_tasks/UR3eAssemblyEnv-v0', render_mode='human')

# Model
model = UR3EBCModel()
# model = UR3EBCRNNModel(True)
# model = UR3EBC3DModel("mc")
# model = UR3EBC3DModelModified("mc")
# model = UR3EFuseEarlyModel("mc")
model = model.to("cuda")
model_path = "/home/students/ur3e_behavioural_cloning/runs/vanilla_no_rnn_final_20250610_203209.pth"
# model_path = "/home/tanakrit-ubuntu/ur3e_behavior_cloning/runs/model_20250409_224215.pth"
model = load_model(model,model_path)

# set up env to replay
env.unwrapped._model =  model


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
    action_exp,state_exp,_, terminated_exp= env.unwrapped.get_action_bt() # expert
    result_state = env.unwrapped.get_action_bt_result()
    action = action 

    if result_state == 1:
        step_limit = 10000
    else:
        step_limit = 5000

    # calculate error
    vel_err.append(np.linalg.norm(action - action_exp))
    hole_err.append(np.linalg.norm(hole.reshape(7) - info["hole_pose"].reshape(7)))
    state_acc.append(1 if state == state_exp else 0)

    observation, reward, terminated, truncated, info = env.step(action)
    step_counter += 1
    # print(step_counter)
    print("State exp: {}, State: {}".format(state_exp, state))
    # print(action)
    
    # reset if state move on
    if result_state != previous_state:
        print("State changed from {} to {}".format(previous_state, result_state))
        time[previous_state] = step_counter
        previous_state = result_state
        
        step_counter = 0

    if step_counter > step_limit or result_state == 4:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()
        eps_counter += 1
        
        # cal avg error
        avg_vel = np.mean(vel_err)
        avg_hole = np.mean(hole_err)
        avg_acc = np.mean(state_acc)


        # update result dict 
        results_dict["states"].append(result_state)
        results_dict["hole_error"].append(avg_hole)
        results_dict["vel_error"].append(avg_vel)
        results_dict["state_acc"].append(avg_acc)
        if time[0] != 0:
            results_dict["time1"].append(time[0])
        if time[1] != 0:
            results_dict["time2"].append(time[1])
        if time[2] != 0:
            results_dict["time3"].append(time[2])
        if time[3] != 0:
            results_dict["time4"].append(time[3])

        # reset all variable
        step_counter = 0
        previous_state = 0
        vel_err = []
        hole_err = []
        state_acc = []
        
        if eps_counter >= test_eps:
            break

# save result dict

# calculate metric
state0_count = results_dict["states"].count(0)
state1_count = results_dict["states"].count(1)
state2_count = results_dict["states"].count(2)
state3_count = results_dict["states"].count(3)
state4_count = results_dict["states"].count(4)

success_rate4 = state4_count / test_eps
success_rate3 = (state4_count+ state3_count) / test_eps
success_rate2 = (state4_count + state3_count + state2_count) / test_eps
success_rate1 = (state4_count + state3_count + state2_count + state1_count) / test_eps

print(results_dict["states"])
time = [0, 0, 0, 0]
for i in range(4):
    if len(results_dict["time" + str(i + 1)]) != 0:
        time[i] = np.mean(results_dict["time" + str(i + 1)])
    
    metric = {
    "success_rate4": success_rate4,
    "success_rate3": success_rate3,
    "success_rate2": success_rate2,
    "success_rate1": success_rate1,
    "avg_hole_error": np.mean(results_dict["hole_error"]),
    "avg_vel_error": np.mean(results_dict["vel_error"]),
    "avg_state_acc": np.mean(results_dict["state_acc"]),
    "avg_time1": time[0]*0.002,
    "avg_time2": time[1]*0.002,
    "avg_time3": time[2]*0.002,
    "avg_time4": time[3]*0.002,
}
print("Metric: ", metric)
# save metric


# Close the environment when the simulation is done
env.close()

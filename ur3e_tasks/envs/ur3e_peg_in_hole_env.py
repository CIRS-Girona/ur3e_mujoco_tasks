import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, AG95
from ur3e_tasks.arenas import PegInHoleArena
from ur3e_tasks.robots import RT2F85
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
from ur3e_tasks.robots import Camera
import cv2

from ur3e_tasks.controllers import EEFVelocityController

class UR3ePegInHoleEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # Define observation space
        # observation_space = [Fx, Fy, Fz, Mx, My, Mz, dx, dy, dz]
        # force and torque limits taken from UR3e datasheet
        self.obs_limit = np.array([30.0, 30.0, 30.0, 10.0, 10.0, 10.0, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(
            low=-self.obs_limit,
            high=self.obs_limit,
            shape=(9,), 
            dtype=np.float64
        )

        # Define action space
        # action_space = [vx, vy, vz, wx, wy] defined in the world frame
        self.act_limit = np.array([0.5, 0.5, 0.5, 0.1, 0.1])
        # self.act_limit = np.array([2.0, 2.0, 2.0, 1.0, 1.0])
        self.action_space = spaces.Box(
            low=-self.act_limit, 
            high=self.act_limit, 
            shape=(5,), 
            dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode
        self.show_cam = False
        ############################
        # create MJCF model
        ############################
        
        # peg in hole areana
        self._arena = PegInHoleArena()

        # mocap target that OSC will try to follow
        self._target = Target(self._arena.mjcf_model)


        ### ur3e arm
        self._arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/ur3e/ur3e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        # Load assembly end effector
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, '..', 'assets','peg-in-hole','peg_ee', 'peg_ee.xml')
        xml_path = os.path.abspath(file_path)
        peg_ee = mjcf.from_path(xml_path)
        self._peg_end = peg_ee.find('body','peg_end')

        # attach EE to arm
        self._arm.attach_tool(peg_ee, pos=[0, 0, 0], quat=[0, 0, 0, 1])
        # move eef_site to the peg tip
        self._arm._eef_site = self._arm._mjcf_root.find('site','peg_ee/peg_end_site')

         # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,1.1], quat=[0.7071068, 0, 0, -0.7071068]
        )


        # Store hole properties
        # self._peg = self._arena.mjcf_model.find('joint', "peg_freejoint")
        self._hole = self._arena.mjcf_model.find('body', "hole")       

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        self._hole_pos = self._physics.bind(self._hole).xpos.copy()

        # Camera 
        self._camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "fixed_camera")
        self._hand_camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")

        # set up controller
        # self._controller = OperationalSpaceController(
        self._controller = EEFVelocityController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kv=200 # TODO: tune this parameter
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None
        self.i = 0

        # more attributes related to rewards computation
        # TODO: define more appropriate values
        self.reward_weights = [1.0,1.0,1.0,1.0,1.0]
        self.clearance = 0.03 # TODO: find out this value, or try to make it dynamically follow the mjcf model
        self.z_threshold = 0.05 # must be very small to make sure the peg is inserted to the hole

        self.max_timestep = 60

    def _get_obs(self) -> np.ndarray:
        # end-effector force-torque
        force = self._physics.data.sensor('ur3e/ee_force').data
        torque = self._physics.data.sensor('ur3e/ee_torque').data
        # position of the hole w.r.t. peg
        peg_end_pos = self._physics.bind(self._peg_end).xpos.copy()
        # NOTE: should I define peg_pos from the joints instead of directly from sim data?
        hole_wrt_peg = self._hole_pos - peg_end_pos
        return np.concatenate((force,torque,hole_wrt_peg))

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        # reset flags
        # self.i = 0

        # reset physics
        with self._physics.reset_context():
            for i in range(500): # give a couple of time to finish reset (~500-2000 steps)
                self.i = self.i +1
                # put arm in a reasonable starting position
                self._physics.bind(self._arm.joints).qpos = [
                    -1.5707,
                    -1.5707,
                    1.5707,
                    -1.5707,
                    -1.5707,
                    0.0,
                ]

                #set gravity to zero
                self._physics.model.opt.gravity = [0,0,0]

                # # put peg into gripper position
                # gripper_pose = self._arm.get_eef_pose(self._physics)
                # gripper_pose[2] = gripper_pose[2] - 0.2
                # self._physics.bind(self._peg).qpos[:3] = gripper_pose[:3]

                # # set gripper to be active to hold the peg
                # self._physics.bind(self._gripper._actuator).ctrl = 250

                self._physics.step()
                if self._render_mode == "human":
                    self._render_frame()
                # time.sleep(0.05)
            
                
            
            # reset gravity back to normal
            self._physics.model.opt.gravity = [0,0,-9.8]

            # # put target in a reasonable starting position
            # hole_pos = self._physics.bind(self._hole).xpos.copy()
            # hole_pos[2] = hole_pos[2] + 0.2
            # self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0, 1])

        # reset flag
        self.i = 0
        
        print("Finish reset !!!")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # flags
        self.i = self.i + 1
        terminated = False
        truncated = False

        # execute action
        # action = [vx, vy, vz, wx, wy]
        # append wz=0 before passing to controller
        target_vel = np.concatenate((action,[0]))

        # # peg in hole testing logic
        # if self.i < 1500:
        #     pass
        # elif self.i < 2500:
        #     hole_pos = self._physics.bind(self._hole).xpos.copy()
        #     hole_pos[2] = hole_pos[2] + 0.1
        #     self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0, 1])
        # else:
        #     terminated = True

        # # set target for ee
        # target_pose = self._target.get_mocap_pose(self._physics)

        # self._controller.run(target_pose)
        
        # # step physics
        # self._physics.step()

        # run velocity controller to move with a target velocity
        # each action is executed 1 second
        for _ in range(10):
            self._controller.run(target_vel)
            # step physics
            self._physics.step()
            time.sleep(0.1)
            # render frame
            if self._render_mode == "human":
                self._render_frame()

        print("i = ", self.i)

        if self.i == self.max_timestep:
            truncated = True
        
        # get observation
        observation = self._get_obs() # return [Fx, Fy, Fz, Mx, My, Mz, dx, dy, dz]

        ## Reward function
        # reward based on distance
        # TODO: refine the values for max_dist
        max_dist = [0.6,0.6,0.5] # xy taken from arena size, z taken from max reach of UR3e
        reward_dist = self.map_reward(observation[-3:],max_dist)

        # reward based on magnitude of action taken
        reward_act = self.map_reward(action,self.act_limit)

        # reward based on contact force
        reward_force = self.map_reward(observation[:6],self.obs_limit[:6])

        # reward (or penalty, actually) based on time step taken
        reward_time = -0.1

        # reward/penalty based on termination
        # task completion is defined based on x-y distance (must be less than the clearance) 
        # and z distance (must be less than a certain threshold)
        task_completed = (np.linalg.norm(observation[-3:-1]) < self.clearance) and (np.abs(observation[-1]) < self.z_threshold)
        # safety violation occurs if any of the detected forces and torques exceeds the limit
        safety_violation = np.any(np.abs(observation[:6]) > self.obs_limit[:6])
        # assign reward and flags
        if task_completed:
            reward_termination = 200
            terminated = True
        elif safety_violation:
            reward_termination = -10
            terminated = True
        else:
            reward_termination = 0

        # compute total reward = weighted average
        reward_list = [reward_dist,reward_act,reward_force,reward_time,reward_termination]
        reward = np.dot(self.reward_weights,reward_list)

        # info = self._get_info()
        
        info = {
            "forces":observation[:3],
            "torques":observation[3:6],
            "distance_to_hole":observation[6:],
            "reward_dist": reward_dist,
            "reward_act": reward_act,
            "reward_force": reward_force,
            "reward_time": reward_time,
            "reward_termination": reward_termination
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
            self._viewer.cam.distance = 1.6
            self._viewer.cam.azimuth = -150
            self._viewer.cam.elevation = -45
            self._viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.824])
            #start rendering camera
            # self._camera._renderer.render()
            

        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()
            # render camera
            if self.show_cam and self.i%10 == 0:
                print("render cam")
                # print(self._camera.image)
                cv2.imshow("fixed_camera",cv2.cvtColor(self._camera.image, cv2.COLOR_RGB2BGR) )
                cv2.imshow("hand_camera",cv2.cvtColor(self._hand_camera.image, cv2.COLOR_RGB2BGR) )
                cv2.waitKey(1)
                # self._camera.shoot()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()

    ############################
    # HELPER FUNCTIONS
    ############################

    def map_reward(self,obs,max):
        '''
            Linearly map observation to its reward in the range of [1,0] based on max value.
            Arguments:
                obs: observation to be mapped
                max: max value for the observation (represents the observation that will receive zero reward)
            Returns:
                reward in the range of [1,0]
        '''
        reward = 1 - np.linalg.norm(obs/max)
        return np.clip(reward,0,1)
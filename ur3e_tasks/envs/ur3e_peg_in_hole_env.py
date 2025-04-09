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
from ur3e_tasks.utils import DomainRandomizer
from manipulator_mujoco.utils.mujoco_utils import get_site_jac
from manipulator_mujoco.utils.controller_utils import pose_error

class UR3ePegInHoleEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # Define observation space
        # observation_space = [Fx, Fy, Fz, Mx, My, Mz, dx, dy, dz]
        # force and torque limits taken from UR3e datasheet
        self.obs_limit = np.array([30.0, 30.0, 30.0, 10.0, 10.0, 10.0, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi])
        # self.obs_limit = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(
            low=-self.obs_limit,
            high=self.obs_limit,
            shape=(12,), 
            dtype=np.float64
        )

        # Define action space
        # action_space = [vx, vy, vz, wx, wy] defined in the world frame
        self.act_limit = np.array([0.2, 0.2, 0.2, 0.1, 0.1])
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

        # set randomizer
        self._randomizer = DomainRandomizer(self._arena._mjcf_model)

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
        self._hole_frame = self._arena.mjcf_model.find('body','hole_frame')

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # store original position of hole
        self._hole_pos_default = self._physics.bind(self._hole).xpos.copy()
        self._hole_quat_default = self._physics.bind(self._hole).xquat.copy()

        # Camera 
        self._camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "fixed_camera")
        self._hand_camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")

        # set up controller
        # self._controller = EEFVelocityController(
        #     physics=self._physics,
        #     joints=self._arm.joints,
        #     eef_site=self._arm.eef_site,
        #     min_effort=-150.0,
        #     max_effort=25.0,
        #     kv=150 # TODO: tune this parameter
        # )

        ###########################################
        # UNCOMMENT THIS PART TO TEST WITH POSITION CONTROLLER
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=0.2,
            vmax_abg=0.5,
        )
        ###########################################

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

        self.dist_threshold = 0.01 # must be very small to make sure the peg is inserted to the hole
        self.max_dist = [0.6,0.6,0.5] # xy taken from arena size, z taken from max reach of UR3e
        self.joint_torque_limits = [54.0,54.0,28.0,9.0,9.0,9.0]

        self.max_timestep = 2500

    def _get_obs(self) -> np.ndarray:
        # end-effector force-torque
        # TODO: check in which frame the values are defined
        sensor_force = self._physics.data.sensor('ur3e/ee_force').data
        print("sensor_force = ", sensor_force)
        sensor_torque = self._physics.data.sensor('ur3e/ee_torque').data
        print("sensor_torque = ", sensor_torque)

        # ## Compute expected internal forces using joint torques and Jacobian
        # attachment_site = self._arm._mjcf_root.find('site','attachment_site')
        # attachment_site_id = self._physics.bind(attachment_site).element_id
        # J = get_site_jac(
        #     self._physics.model.ptr, 
        #     self._physics.data.ptr, 
        #     attachment_site_id,
        # )
        # print("J = ", J)

        # R_world_to_sensor = self._physics.bind(attachment_site).xmat.reshape(3,3)
        # print("R_world_to_sensor = ", R_world_to_sensor)

        # tau = self._physics.data.qfrc_passive + self._physics.data.qfrc_bias
        # print("tau = ",tau)

        # expected_force = np.linalg.pinv(J.T) @ tau # in world frame
        # # Transform internal force to sensor frame
        # F_int_sensor = R_world_to_sensor.T @ expected_force[:3]
        # T_int_sensor = R_world_to_sensor.T @ expected_force[3:]

        # print("internal force in sensor frame = ", F_int_sensor)
        # print("internal torque in sensor frame = ", T_int_sensor)

        # external_force = sensor_force - F_int_sensor
        # external_torque = sensor_torque - T_int_sensor


        # position of the hole w.r.t. peg
        peg_end_pos = self._physics.bind(self._peg_end).xpos.copy()
        # NOTE: should I define peg_pos from the joints instead of directly from sim data?
        # hole_wrt_peg_pos = self._hole_pos - peg_end_pos

        # orientation of the hole w.r.t. peg
        peg_end_quat = self._physics.bind(self._peg_end).xquat.copy()
        peg_end_quat_xyzw = [peg_end_quat[1], peg_end_quat[2], peg_end_quat[3], peg_end_quat[0]]
        # print("peg end quat = ", peg_end_quat)
        # hole_wrt_peg_quat = orientation_error(quat2mat(self._hole_quat), quat2mat(peg_end_quat))

        ## alternative: directly calculate pose difference
        peg_end_pose = np.concatenate((peg_end_pos,peg_end_quat_xyzw))
        hole_frame_pose = np.concatenate((self._hole_pos, self._hole_quat_xyzw))
        hole_wrt_peg_pose = pose_error(hole_frame_pose,peg_end_pose)
        print("hole_wrt_peg_pose = ", hole_wrt_peg_pose)
        return np.concatenate((sensor_force,sensor_torque,hole_wrt_peg_pose))

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos = [
                -1.5707,
                -1.5707,
                1.5707,
                -1.5707,
                -1.5707,
                0.0,
            ]

            # randomize hole position and orientation
            rand_pos = self._randomizer.get_random_ws_pos()
            self._physics.bind(self._hole).mocap_pos[:] = rand_pos

            rand_quat = self._randomizer.get_random_quat(self._hole_quat_default)
            self._physics.bind(self._hole).mocap_quat[:] = rand_quat

            # update physics with the randomized position
            self._physics.forward()

            # store the randomized position of the hole (for observation)
            self._hole_pos = self._physics.bind(self._hole_frame).xpos.copy()
            print("hole pos after reset = ", self._hole_pos)
            self._hole_quat = self._physics.bind(self._hole_frame).xquat.copy() # format: wxzy
            self._hole_quat_xyzw = [self._hole_quat[1], self._hole_quat[2], self._hole_quat[3], self._hole_quat[0]]
            
            # reset gravity back to normal
            self._physics.model.opt.gravity = [0,0,-9.8]

            ###########################################
            # UNCOMMENT THIS PART TO TEST WITH POSITION CONTROLLER
            # put target in a reasonable starting position
            target_pos = self._hole_pos.copy()
            R_world_to_hole = self._physics.bind(self._hole_frame).xmat.reshape(3,3)
            offset = R_world_to_hole @ np.array([0,0,0.2]).T
            print("offset = ",offset)
            target_pos += offset
            self._target.set_mocap_pose(self._physics, position=target_pos[:3], quaternion=self._hole_quat_xyzw.copy())
            ############################################


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

        ###########################################
        # UNCOMMENT THIS PART TO TEST WITH POSITION CONTROLLER
        # peg in hole testing logic
        if self.i < 500:
            pass
        elif self.i < 2500:
            hole_pos = self._physics.bind(self._hole_frame).xpos.copy()
            hole_pos[2] = hole_pos[2]
            target_quat = [self._hole_quat.copy()[1], self._hole_quat.copy()[2], self._hole_quat.copy()[3], self._hole_quat.copy()[0]]
            self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=target_quat)
        else:
            terminated = True

        # set target for ee
        target_pose = self._target.get_mocap_pose(self._physics)
        ###########################################

        # run velocity controller to move with a target velocity
        # each action is executed 10 times before getting new observation
        for _ in range(10):
            self._controller.run(target_pose) # CHANGE TO target_vel TO USE VELOCITY CONTROLLER
            # step physics
            self._physics.step()
            #time.sleep(0.01)
            # render frame
            if self._render_mode == "human":
                self._render_frame()

        print("i = ", self.i)

        if self.i == self.max_timestep:
            truncated = True
        
        # get observation
        observation = self._get_obs() # return [Fx, Fy, Fz, Mx, My, Mz, dx, dy, dz]

        ## Reward function
        reward, terminated, reward_list = self._get_reward(observation,action)

        # info = self._get_info()
        
        info = {
            "forces":observation[:3],
            "torques":observation[3:6],
            "distance_to_hole":observation[6:9],
            "orientation_difference":observation[9:],
            "reward_distance": reward_list[0],
            "reward_action": reward_list[1],
            "reward_force": reward_list[2],
            "reward_time": reward_list[3],
            "reward_termination": reward_list[4]
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

    def _get_reward(self,observation,action):
        ## Reward function
        # reward based on distance
        reward_dist = self.map_reward(observation[6:9],self.max_dist)
        print("reward_dist = ",reward_dist)

        # reward based on magnitude of action taken
        reward_act = self.map_reward(action,self.act_limit)
        print("reward_act = ",reward_act)

        # reward based on contact force
        reward_force = self.map_reward(observation[:6],self.obs_limit[:6])
        print("reward_force = ",reward_force)

        # reward (or penalty, actually) based on time step taken
        reward_time = -0.1
        print("reward_time = ",reward_time)

        # reward/penalty based on termination
        # task completion is defined based on x-y distance (must be less than the clearance) 
        # and z distance (must be less than a certain threshold)
        # TODO: modify task_completed to comply with random rotations (now it's still in world frame!)
        # task_completed = (np.linalg.norm(observation[6:8]) < self.clearance) and (np.abs(observation[8]) < self.z_threshold)
        
        # task completion is defined based on distance between hole and peg (must be less than a certain threshold)
        print("distance from hole = ", np.linalg.norm(observation[6:9]))
        task_completed = np.linalg.norm(observation[6:9]) < self.dist_threshold

        # safety violation occurs if any of the detected forces and torques exceeds the limit
        safety_violation = self.check_safety_violation(observation[:6])

        # assign reward and flags
        if task_completed:
            reward_termination = 200
            terminated = True
        elif safety_violation:
            reward_termination = -10
            terminated = True
        else:
            reward_termination = 0
            terminated = False

        print("reward_termination = ", reward_termination)

        # compute total reward = weighted average
        reward_list = [reward_dist,reward_act,reward_force,reward_time,reward_termination]
        reward = np.dot(self.reward_weights,reward_list)

        return reward, terminated, reward_list

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
    
    def check_safety_violation(self,ee_force_torque):
        '''
            Returns True if safety violation occurs.
            Safety violation is defined by one of these conditions:
            * end-effector (or tool flange) force-torque sensor reading in any axis exceeds its limit,
            * torque in any of the joints exceeds its limit.
        '''
        # end-effector force-torque
        ee_safety_violation = np.any(np.abs(ee_force_torque) >= self.obs_limit[:6])

        # extract joint torques
        qfrc_bias = self._physics.data.qfrc_bias
        qfrc_passive = self._physics.data.qfrc_passive
        qfrc_applied = self._physics.data.qfrc_applied

        total_joint_torques = qfrc_bias + qfrc_passive + qfrc_applied

        joint_safety_violation = np.any(np.abs(total_joint_torques) >= self.joint_torque_limits)

        return ee_safety_violation or joint_safety_violation
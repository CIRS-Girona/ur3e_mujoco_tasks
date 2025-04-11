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
from manipulator_mujoco.utils.transform_utils import mat2quat

class UR3ePegInHoleEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # Define observation space
        # observation_space = [end-effector force and torque, pose of hole w.r.t. peg, joint positions]
        # force and torque limits taken from UR3e datasheet
        self.obs_limit = np.array([30.0, 30.0, 30.0, # force limits
                                   10.0, 10.0, 10.0, # torque limits
                                   np.inf, np.inf, np.inf, # distance limits
                                   1.0, 1.0, 1.0, 1.0, # angular difference (quat) limits
                                   np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) # joint position limits
        self.observation_space = spaces.Box(
            low=-self.obs_limit,
            high=self.obs_limit,
            shape=(19,), 
            dtype=np.float64
        )

        # Define action space
        # action_space = [vx, vy, vz, wx, wy] defined in the world frame
        self.act_limit = np.array([0.2, 0.2, 0.2, 0.1, 0.1])
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
        self._arm.attach_tool(peg_ee, pos=[0, 0, 0], quat=[1, 0, 0, 0])
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
        # TODO: tune these values
        self.reward_weights = [1.0,0.05,0.4] # [distance, action, force]
        self.dist_threshold = 0.01 # must be very small to make sure the peg is inserted to the hole
        self.max_dist = [0.6,0.6,0.5] # xy taken from arena size, z taken from max reach of UR3e
        self.joint_torque_limits = [54.0,54.0,28.0,9.0,9.0,9.0]

        self.max_timestep = 2500

    def _get_obs(self) -> np.ndarray:
        ## end-effector force-torque
        sensor_force = self._physics.data.sensor('ur3e/ee_force').data.copy()
        print("sensor_force = ", sensor_force)
        sensor_torque = self._physics.data.sensor('ur3e/ee_torque').data.copy()
        print("sensor_torque = ", sensor_torque)

        ########################################################
        # # position of the hole w.r.t. peg
        # peg_end_pos = self._physics.bind(self._peg_end).xpos.copy()
        # # NOTE: should I define peg_pos from the joints instead of directly from sim data?
        # # hole_wrt_peg_pos = self._hole_pos - peg_end_pos

        # # orientation of the hole w.r.t. peg
        # peg_end_quat = self._physics.bind(self._peg_end).xquat.copy()
        # peg_end_quat_xyzw = [peg_end_quat[1], peg_end_quat[2], peg_end_quat[3], peg_end_quat[0]]
        # # print("peg end quat = ", peg_end_quat)
        # # hole_wrt_peg_quat = orientation_error(quat2mat(self._hole_quat), quat2mat(peg_end_quat))

        # ## alternative: directly calculate pose difference
        # peg_end_pose = np.concatenate((peg_end_pos,peg_end_quat_xyzw))
        # hole_frame_pose = np.concatenate((self._hole_pos, self._hole_quat_xyzw))
        # hole_wrt_peg_pose = pose_error(hole_frame_pose,peg_end_pose)
        # print("hole_wrt_peg_pose = ", hole_wrt_peg_pose)
        ############################################################################
        
        ## position and orientation of hole w.r.t. peg
        # position and orientation of peg tip w.r.t. world
        peg_end_pos = self._physics.bind(self._peg_end).xpos.copy()
        peg_end_rot = self._physics.bind(self._peg_end).xmat.copy() # rotation matrix
        peg_end_rot = peg_end_rot.reshape(3,3)
        peg_end_transform = np.block([[peg_end_rot,peg_end_pos.reshape(-1,1)],[0,0,0,1]]) # transformation matrix

        # position and orientation of hole w.r.t. world
        hole_transform = np.block([[self._hole_rot,self._hole_pos.reshape(-1,1)],[0,0,0,1]]) # transformation matrix

        # multiply transform matrices to obtain transformation from peg to hole
        hole_wrt_peg_transform = np.linalg.inv(peg_end_transform) @ hole_transform

        # extract position and orientation of hole w.r.t. peg
        hole_wrt_peg_pos = hole_wrt_peg_transform[:3,3]
        hole_wrt_peg_rot = hole_wrt_peg_transform[:3,:3] # rotation matrix
        hole_wrt_peg_quat = mat2quat(hole_wrt_peg_rot) # format: xyzw

        print("hole_wrt_peg_pos = ", hole_wrt_peg_pos)
        print("hole_wrt_peg_quat = ", hole_wrt_peg_quat)

        ## joint positions
        joint_pos = self._physics.data.qpos.copy()
        print("joint pos = ",joint_pos)
        return np.concatenate((sensor_force,sensor_torque,hole_wrt_peg_pos,hole_wrt_peg_quat,joint_pos))

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

            # store the randomized position and orientation of the hole (for observation)
            self._hole_pos = self._physics.bind(self._hole_frame).xpos.copy()
            print("hole pos after reset = ", self._hole_pos)
            self._hole_quat = self._physics.bind(self._hole_frame).xquat.copy() # format: wxzy
            self._hole_quat_xyzw = [self._hole_quat[1], self._hole_quat[2], self._hole_quat[3], self._hole_quat[0]]
            self._hole_rot = self._physics.bind(self._hole_frame).xmat.copy()
            self._hole_rot = self._hole_rot.reshape(3,3)
            
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
        observation = self._get_obs()

        ## Reward function
        reward, terminated, reward_list = self._get_reward(observation,action)

        # info = self._get_info()
        
        info = {
            "forces":observation[:3],
            "torques":observation[3:6],
            "distance_to_hole":observation[6:9],
            "orientation_difference":observation[9:12],
            "joint_pos":observation[12:],
            "reward_distance": reward_list[0],
            "reward_action": reward_list[1],
            "reward_force": reward_list[2],
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

        reward_list = [reward_dist,reward_act,reward_force]

        # reward/penalty based on termination        
        # task completion is defined based on distance between hole and peg
        # (must be less than a certain threshold)
        print("distance from hole = ", np.linalg.norm(observation[6:9]))
        task_completed = np.linalg.norm(observation[6:9]) < self.dist_threshold

        # safety violation occurs if any of the detected forces and torques exceeds the limit
        safety_violation = self.check_safety_violation(observation[:6])

        # assign reward and flags
        if task_completed:
            reward = 100
            terminated = True
        elif safety_violation:
            reward = -100
            terminated = True
        else:
            reward = np.dot(self.reward_weights,reward_list) # weighted combination
            terminated = False

        return reward, terminated, reward_list

    ############################
    # HELPER FUNCTIONS
    ############################

    def map_reward(self,vec,max):
        '''
            Normalize vec based on max value, and assign negative reward.
            In general, higher value on vec means lower reward.
            Arguments:
                vec: observation/action to be mapped
                max: max value for the vec
            Returns:
                reward (always negative)
        '''
        reward = - np.linalg.norm(vec/max)
        return reward
    
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
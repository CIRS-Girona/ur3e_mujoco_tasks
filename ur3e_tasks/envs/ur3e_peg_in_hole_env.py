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
from manipulator_mujoco.utils.transform_utils import mat2quat, quat2axisangle, quat2mat, axisangle2quat
from ur3e_tasks.utils.curriculum import CurriculumLearning

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
                                   np.inf, np.inf, np.inf, # peg position limits
                                   1.0, 1.0, 1.0, 1.0, # peg quat limits
                                   np.inf, np.inf, np.inf, # hole position limits
                                   1.0, 1.0, 1.0, 1.0, # hole quat limits
                                   np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) # joint position limits
        self.observation_space = spaces.Box(
            low=-self.obs_limit,
            high=self.obs_limit,
            shape=(26,), 
            dtype=np.float64
        )

        # Define action space
        # action_space = [vx, vy, vz, wx, wy] defined in the world frame
        # self.act_limit = np.array([0.2, 0.2, 0.2, 0.1, 0.1])
        self.act_limit = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
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

        # Initialize visualization of intermediate point
        self.intermediate_target = self._arena.mjcf_model.worldbody.add("body", name="intermediate_pt", mocap=True)
        self.intermediate_target.add(
                "geom",
                type="sphere",
                size=[0.01],
                rgba=[1, 0, 0, 0.2],
                conaffinity=0,
                contype=0,
                group=2
            )
        
        ######################
        
        # # generate model
        # self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # # store original position of hole
        # self._hole_pos_default = self._physics.bind(self._hole).xpos.copy()
        # self._hole_quat_default = self._physics.bind(self._hole).xquat.copy()

        # # Camera 
        # # self._camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "fixed_camera")
        # # self._hand_camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")

        # # set up controller
        # self.joint_torque_limits = np.array([54.0,54.0,28.0,9.0,9.0,9.0])
        # self._controller = EEFVelocityController(
        #     physics=self._physics,
        #     joints=self._arm.joints,
        #     eef_site=self._arm.eef_site,
        #     min_effort=-self.joint_torque_limits,
        #     max_effort=self.joint_torque_limits,
        #     kv=120 # TODO: tune this parameter
        # )

        # # for GUI and time keeping
        # self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        # self._step_start = None
        # self.i = 0

        ######################

        # compile model now if visualization is not needed
        # (to avoid creating new physics over and over again)
        if render_mode != "human":
            self.compile_model()

        # more attributes related to rewards computation
        # TODO: tune these values
        self.reward_weights = [1.5,0.0,1.0] # [distance, action, force]
        self.max_dist = [0.6,0.6,0.5] # xy taken from arena size, z taken from max reach of UR3e

        # attribute related to curriculum learning
        self.curriculum = CurriculumLearning()
        self.learning_stage = 1
        self.prev_learning_stage = 0

        # attribute related to hole position noise
        self.pos_noise = 0.005


    def _get_obs(self) -> np.ndarray:
        ## end-effector force-torque
        sensor_force = self._physics.data.sensor('ur3e/ee_force').data.copy()
        sensor_torque = self._physics.data.sensor('ur3e/ee_torque').data.copy()
        
        ## position and orientation of peg (w.r.t. robot base)
        self._peg_end_pos = self._physics.bind(self._peg_end).xpos.copy()

        peg_end_rot = self._physics.bind(self._peg_end).xmat.copy() # rotation matrix
        self._peg_end_rot = peg_end_rot.reshape(3,3)

        # peg_end_quat = self._physics.bind(self._peg_end).xquat.copy() #wxyz

        self._peg_end_pos_base, self._peg_end_quat_base = self.convert_to_base_frame(pos=self._peg_end_pos.copy(),
                                                                                     rot = self._peg_end_rot.copy(),
                                                                                     return_quat=True)

        ## joint positions
        joint_pos = self._physics.data.qpos.copy()

        return np.concatenate((sensor_force, 
                               sensor_torque, 
                               self._peg_end_pos_base,
                               self._peg_end_quat_base,
                               self._hole_pos_base_obs,
                               self._hole_quat_base,
                               joint_pos))

    def _get_info(self) -> dict:
        # only called at reset
        return {
            "learning_stage":self.learning_stage,
            "intermediate_target_pos":self.intermediate_target_pos}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        if seed is not None:
            self._randomizer.set_seed(seed)

        # update visualization of intermediate point if learning stage change
        if self.learning_stage != self.prev_learning_stage and self._render_mode == "human":
            # get distance threshold to update visualization
            self.dist_threshold = self.curriculum.get_distance_threshold(self.learning_stage)
            # reset visualization of intermediate point
            for geom in list(self.intermediate_target.find_all('geom')):
                geom.size = [self.dist_threshold]

            # (re-)generate physics
            self.compile_model()

            self.prev_learning_stage = self.learning_stage


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
            self._hole_rot = self._physics.bind(self._hole_frame).xmat.copy()
            self._hole_rot = self._hole_rot.reshape(3,3)
            # self._hole_quat = self._physics.bind(self._hole_frame).xquat.copy() # format: wxzy
            # self._hole_quat_xyzw = [self._hole_quat[1], self._hole_quat[2], self._hole_quat[3], self._hole_quat[0]]

            # transform hole frame pose from world frame to robot base frame
            self._hole_pos_base, self._hole_quat_base = self.convert_to_base_frame(pos=self._hole_pos.copy(),
                                                                                  rot=self._hole_rot.copy(),
                                                                                  return_quat=True)
            
            # add position noise to the hole
            if self.curriculum.generate_noise_flag(self.learning_stage):
                pos_noise = self.generate_position_noise(self.pos_noise)
                self._hole_pos_obs = self._hole_pos.copy() + self._hole_rot @ pos_noise
                # convert to robot base frame
                self._hole_pos_base_obs, _ = self.convert_to_base_frame(pos=self._hole_pos_obs.copy(),
                                                                    rot=self._hole_rot.copy())
            else:
                self._hole_pos_obs = self._hole_pos.copy()
                self._hole_pos_base_obs = self._hole_pos_base.copy()

            ######################################################
            # # # debugging
            # print("hole_pos", self._hole_pos)
            # # print("pos_noise", pos_noise)
            # print("hole_pos_obs", self._hole_pos_obs)

            # print("hole_pos_base", self._hole_pos_base)
            # print("hole_pos_base_obs", self._hole_pos_base_obs)
            ######################################################
            
            # reset gravity back to normal
            self._physics.model.opt.gravity = [0,0,-9.8]

            # store initial peg end position and orientation
            self._peg_end_pos = self._physics.bind(self._peg_end).xpos.copy()
            peg_end_rot = self._physics.bind(self._peg_end).xmat.copy() # rotation matrix
            self._peg_end_rot = peg_end_rot.reshape(3,3)

            # set intermediate target position and distance threshold based on curriculum
            self.intermediate_target_pos = self.curriculum.get_target_point(self._hole_pos.copy(),
                                                                            self._hole_rot.copy(),
                                                                            self.learning_stage)   

            # Update visualization of intermediate target position
            self.intermediate_target = self._arena.mjcf_model.find("body","intermediate_pt")
            self._physics.bind(self.intermediate_target).mocap_pos[:] = self.intermediate_target_pos
            

        # reset flag
        self.i = 0
        
        print("Finish reset !!!")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def compile_model(self):
        self.close()

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # store original position of hole
        self._hole_pos_default = self._physics.bind(self._hole).xpos.copy()
        self._hole_quat_default = self._physics.bind(self._hole).xquat.copy()

        # set up controller
        self.joint_torque_limits = np.array([54.0,54.0,28.0,9.0,9.0,9.0])
        self._controller = EEFVelocityController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-self.joint_torque_limits,
            max_effort=self.joint_torque_limits,
            kv=120 # TODO: tune this parameter
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None
        self.i = 0

    def step(self, action: np.ndarray) -> tuple:
        # flags
        self.i = self.i + 1
        terminated = False
        truncated = False # always false; will be taken care by the `TimeLimit` wrapper added during `make`

        # execute action
        # action = [vx, vy, vz, wx, wy] in end-effector frame
        # append wz=0 before passing to controller
        target_vel_ee = np.concatenate((action,[0]))

        # convert target vel to world frame
        target_vel = self.convert_twist_to_world(target_vel_ee)

        # run velocity controller to move with a target velocity
        # each action is executed 10 times before getting new observation
        for _ in range(25):
            self._controller.run(target_vel)
            # step physics
            self._physics.step()
            # render frame
            if self._render_mode == "human":
                self._render_frame()

        # print("i = ", self.i)
        
        # get observation
        observation = self._get_obs()

        ## Reward function
        reward, terminated, reward_list, success = self._get_reward(observation,action)

        # print(f"action = {action}")
        # print(f"observation = {observation}")
        # print(f"reward = {reward}")
        
        info = {
            "learning_stage":self.learning_stage,
            "intermediate_target_pos":self.intermediate_target_pos,
            "forces":observation[:3],
            "torques":observation[3:6],
            "peg_end_pos":observation[6:9],
            "peg_end_quat":observation[9:13],
            "hole_pos":observation[13:16],
            "hole_quat":observation[16:20],
            "joint_pos":observation[20:],
            "reward_distance": reward_list[0],
            "reward_action": reward_list[1],
            "reward_force": reward_list[2],
            "is_success": success
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

            # --- Enable contact point and force visualization ---
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            

        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

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
        distance = self._hole_pos.copy() - self._peg_end_pos.copy()
        reward_dist = self.map_reward(distance,self.max_dist)

        # reward based on magnitude of action taken
        reward_act = self.map_reward(action,self.act_limit)

        # reward based on contact force
        # reward (or penalty) is a step function at the force limit
        ee_safety_violation = np.any(np.abs(observation[:6]) >= self.obs_limit[:6])
        if ee_safety_violation:
            print(f"Contact force exceeds limit! {observation[:6]}")
            reward_force = self.curriculum.get_force_penalty(self.learning_stage)
        else:
            reward_force = 0.0

        reward_list = [reward_dist,reward_act,reward_force]

        # reward/penalty based on termination        
        # task completion is defined based on the learning stage
        task_completed = self.curriculum.check_task_completed(self._peg_end_pos.copy(),
                                                              self._peg_end_rot.copy(),
                                                              self._hole_pos.copy(),
                                                              self._hole_rot.copy(),
                                                              self.learning_stage)

        # safety violation occurs if any of the joint torques exceeds the limit
        safety_violation = self.check_safety_violation()

        # assign reward and flags
        terminated = False
        success = False # flag to indicate episode is successful
        if task_completed:
            reward = 100
            success = True
            terminated = True
        elif safety_violation:
            reward = -20
        else:
            reward = np.dot(self.reward_weights,reward_list) # weighted combination

        return reward, terminated, reward_list, success
    
    def set_learning_stage(self,stage):
        '''
            Set learning stage.
            Args:
                stage: (int) learning stage to set.
        '''
        self.learning_stage = stage
        print(f"Setting environment to stage {self.learning_stage}")

    ############################
    # HELPER FUNCTIONS
    ############################

    def map_reward(self,vec,max):
        '''
            Normalize vec based on max value, and assign negative reward.
            In general, higher value on vec means lower reward.
            
            Args:
                vec: observation/action to be mapped
                max: max value for the vec
            
            Returns:
                reward (always negative)
        '''
        reward = - np.linalg.norm(vec/max)
        return reward
    
    def check_safety_violation(self):
        '''
            Returns True if safety violation occurs: torque in any of the joints exceeds its limit.
        '''
        # extract joint torques
        qfrc_passive = self._physics.data.qfrc_passive # passive forces from spring-dampers and fluid dynamics
        qfrc_applied = self._physics.data.qfrc_applied # applied by the controller

        total_joint_torques = qfrc_passive + qfrc_applied
        # print(f"total joint torques = {total_joint_torques}")        

        joint_safety_violation = np.any(np.abs(total_joint_torques) >= self.joint_torque_limits)

        return joint_safety_violation
    
    def convert_twist_to_world(self,twist_b):
        '''
            Convert twist from peg frame to world frame.
            
            Args:
                twist_b: (ndarray) twist in peg frame.
            
            Returns:
                twist expressed in world frame.
        '''
        # adjoint matrix
        R = self._peg_end_rot
        p = self._peg_end_pos
        p_ss = np.array([[0,-p[2],p[1]],
                         [p[2],0,-p[0]],
                         [-p[1],p[0],0]])
        A = np.block([[R, np.zeros((3,3))],[np.zeros((3,3)),R]])
        
        twist_w = A @ twist_b.reshape(-1,1)
        return twist_w.reshape(-1)
    
    def generate_position_noise(self,distance):
        '''
            Generate noise to the hole position (to be compounded with the original position).
            
            Args:
                distance: (float) radius of noise to be added.
            
            Returns:
                offset: (ndarray) noise to be added to the hole position.
        '''
        random_angle = np.random.uniform(0, 2 * np.pi)
        offset = np.array([
            distance * np.cos(random_angle),
            distance * np.sin(random_angle),
            0.0
        ])
        return offset
    
    def convert_to_base_frame(self,pos,rot=None,quat=None,return_quat=False):
        '''
            Convert pose from world frame to robot base frame.
            Orientation can be given in either rotation matrix or quaternion form (but not both).
            
            Args:
                pos: (ndarray) position w.r.t. world frame
                rot: (ndarray) 3x3 rotation matrix indicating orientation w.r.t. world frame.
                quat: (ndarray) quaternion in xyzw format indicating orientation w.r.t. world frame.
                return_quat: (bool) if True, orientation is returned in quaternion (xyzw format); otherwise, orientation is returned in 3x3 rotation matrix.
            
            Returns:
                position and orientation w.r.t. robot base frame.
        '''
        assert (rot is None and quat is not None) or (rot is not None and quat is None), "Between rot and quat, only 1 can be given"

        # access transformation from world to robot base
        base_id = self._arena.mjcf_model.find('body','ur3e/base')
        base_pos = self._physics.bind(base_id).xpos.copy()
        base_rot = self._physics.bind(base_id).xmat.copy()
        base_rot = base_rot.reshape(3,3)
        base_transform = np.block([[base_rot,base_pos.reshape(-1,1)],[0,0,0,1]])

        # construct transformation of the given pose w.r.t. world
        if quat is not None:
            rot = quat2mat(quat)
        pose_transform = np.block([[rot,pos.reshape(-1,1)],[0,0,0,1]])

        # compound transformation
        pose_transform_base = np.linalg.inv(base_transform) @ pose_transform

        # extract position and rotation
        pos_wrt_base = pose_transform_base[:3,3]
        rot_wrt_base = pose_transform_base[:3,:3]

        if return_quat:
            quat_wrt_base = mat2quat(rot_wrt_base)
            return pos_wrt_base,quat_wrt_base
        else:
            return pos_wrt_base,rot_wrt_base

import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.robots import Arm
from ur3e_tasks.arenas import AssemblyArena
from ur3e_tasks.robots import Suction, RT2F85

from ur3e_tasks.utils import  DomainRandomizer
from ur3e_tasks.utils import  AssemblyBT

from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
from ur3e_tasks.robots import Camera
import cv2

class UR3eAssemblyEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # TODO come up with an observation space that makes sense
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )

        # TODO come up with an action space that makes sense
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._viewer = None
        self._render_mode = render_mode
        self.show_cam = False
        ############################
        # create MJCF model
        ############################
        
        # peg in hole areana
        self._arena = AssemblyArena()

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


        
        # Get Hole position
        self._hole = self._arena.mjcf_model.find('site', "hole_site")
        

        # # Get Peg position
        # self._peg_pickup = self._arena.mjcf_model.find('site', "peg_pickup")
        # self._peg = self._arena.mjcf_model.find('site', "peg_base")
        # self._peg_joint = self._arena.mjcf_model.find('joint', "peg_freejoint")


        # Load assembly end effector
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, '..', 'assets','assembly','assembly_ee', 'assembly_ee.xml')
        xml_path = os.path.abspath(file_path)
        assembly_ee = mjcf.from_path(xml_path)

        # attach EE to arm
        self._arm.attach_tool(assembly_ee, pos=[0, 0, 0], quat=[0, 0, 0, 1])
        # move eef_site to end effector tip
        self._arm._eef_site = self._arm._mjcf_root.find("site","assembly_ee/tip_site")
        

        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,1], quat=[0.7071068, 0, 0, -0.7071068]
        )


        self._randomizer = DomainRandomizer(self._arena._mjcf_model)

        # self._weld = self._gripper.setup_weld(self._arena.mjcf_model,"peg" )
        # self.complie_model()
        
        # # generate model
        # self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # # Camera 
        # self._camera = Camera([480, 320], self._physics.model.ptr, self._physics.data.ptr, "fixed_camera")
        # # self._hand_camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")

        # # set up OSC controller
        # self._controller = OperationalSpaceController(
        #     physics=self._physics,
        #     joints=self._arm.joints,
        #     eef_site=self._arm.eef_site,
        #     min_effort=-150.0,
        #     max_effort=150.0,
        #     kp=200,
        #     ko=200,
        #     kv=50,
        #     vmax_xyz=1.0,
        #     vmax_abg=2.0,
        # )

        # # for GUI and time keeping
        # self._timestep = self._physics.model.opt.timestep
        # self._viewer = None
        # self._step_start = None
        # self.i = 0

    def _get_obs(self) -> np.ndarray:
        # TODO come up with an observations that makes sense for your RL task
        return np.zeros(6)

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        # reset flags
        self.i = 0
        # domain randomize
        self._randomizer.random_object_in_ws("hole")
        self._randomizer.random_texture("table_top")
        self._randomizer.random_texture("wall_left")
        self._randomizer.random_texture("wall_right")
        self._randomizer.random_texture("wall_back")
        self._randomizer.random_light("light_source")
        self._randomizer.random_object_color("hole")
        self._randomizer.random_object_color("ur3e/assembly_ee/peg_ee")
        self._randomizer.random_object_color("ur3e/assembly_ee/peg_ee_base")
        self._randomizer.random_arm_height("ur3e/base")
        self._randomizer.random_fixed_camera("fixed_camera","camera_center")
        self._randomizer.random_distractors()
        self._randomizer.random_texture_arm(self._arm)
        
        # recomplie model
        self.complie_model()

        # find initial arm position
        init_pos = self._randomizer.get_random_ws_pos_clearance(self._physics.bind(self._hole).xpos.copy(), 0.3)
        init_quat = self._randomizer.random_quaternion_around_axis([0, 0, 0, 1],['x','y','z'], np.pi/8)
        self._target.set_mocap_pose(self._physics, position=init_pos, quaternion=init_quat)
        target_pose = self._target.get_mocap_pose(self._physics)
        
        # reset physics
        with self._physics.reset_context():
            self._physics.bind(self._arm.joints).qpos = [
                    -1.5707,
                    -1.5707,
                    1.5707,
                    -1.5707,
                    -1.5707,
                    0.0,
                ]
            
            for i in range(1000): # give a couple of time to move to initial position (~500-2000 steps)
                self.i = self.i +1
                # put arm in a reasonable starting position
                

                # random initial position
                self._controller.run(target_pose)

            #     self._physics.step()
                # if self._render_mode == "human":
                #     self._render_frame()
          

        
        print("Finish reset !!!")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # flags
        self.i = self.i + 1
        terminated = False

        # behavior
        cmd, state, success, terminated = self._bt.run()
        # print('-------------------------')
        # print(f"Command: {cmd}")
        # print(f"State: {state}")
        # print(f"Success: {success}")
        # print(f"Terminated: {terminated}")
        # print('-------------------------')

        
        # # peg in hole testing logic
        # if self.i < 2500: # Move peg to a position on top of peg
        #     hole_pos = self._physics.bind(self._hole).xpos.copy()
        #     hole_pos[2] = hole_pos[2] + 0.1
        #     self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0, 1])
        # elif self.i < 4000: # assemble
        #     hole_pos = self._physics.bind(self._hole).xpos.copy()
        #     hole_pos[2] = hole_pos[2] + 0.002
        #     self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0, 1])
        # elif self.i < 5000: # rotate peg inside the hole
        #     hole_pos = self._physics.bind(self._hole).xpos.copy()
        #     hole_pos[2] = hole_pos[2] + 0.002
        #     self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0.7071068, 0.7071068])
        # else:
        #     terminated = True
        if cmd is not None:
            self._target.set_mocap_pose(self._physics, position=cmd[:3], quaternion=cmd[3:])
            # set target for ee
            target_pose = self._target.get_mocap_pose(self._physics)

            # run OSC controller to move to target pose
            self._controller.run(target_pose)

        # step physics
        self._physics.step()
        # time.sleep(0.01)

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        # terminated = False
        info = self._get_info()


        return observation, reward, terminated, False, info

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
            self._viewer.cam.distance = 1.2
            self._viewer.cam.azimuth = -150
            self._viewer.cam.elevation = -45
            self._viewer.cam.lookat[:] = np.array([0.0, 0.0, 1.0])
            # start rendering camera
            # self._camera._renderer.render()
            print("Start viewer")
            

        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()
            

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()
            # render camera
            if self.show_cam and self.i%20 == 0:
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
            self._camera._renderer.close()
            self._hand_camera._renderer.close()

    def complie_model(self):
        self.close()

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # Camera 
        self._camera = Camera([480, 320], self._physics.model.ptr, self._physics.data.ptr, "fixed_camera")
        self._hand_camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")
        
        # self._hand_camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")

        # set up OSC controller
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # set up behavior
        self._bt = AssemblyBT(self._physics,self._arm.eef_site,self._hole)

        # for GUI and time keeping
        self._viewer = None
        self._timestep = self._physics.model.opt.timestep
        self._step_start = None
        self.i = 0
        print("Finish Compiling")
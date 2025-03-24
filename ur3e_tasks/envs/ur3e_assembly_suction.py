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
        self.show_cam = True
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
        self._hole_joint = self._arena.mjcf_model.find('joint', "hole_joint")
        self._hole_body = self._arena.mjcf_model.find('body', "hole")
        

        # Get Peg position
        self._peg_pickup = self._arena.mjcf_model.find('site', "peg_pickup")
        self._peg = self._arena.mjcf_model.find('site', "peg_base")
        self._peg_joint = self._arena.mjcf_model.find('joint', "peg_freejoint")


        # ag95 gripper
        self._gripper = Suction()
        # attach EE to arm
        self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[0, 0, 0, 1])
        

        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,1], quat=[0.7071068, 0, 0, -0.7071068]
        )

        # connect arm to mocap object
        self._arena._mjcf_model.equality.add("weld",name="arm_connect",body1="base_anchor",body2="ur3e/base",anchor="0 0 0 ",active="true")


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
        # recomplie model
        self.complie_model()
        # reset physics
        with self._physics.reset_context():
            for i in range(10): # give a couple of time to finish reset (~500-2000 steps)
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
                
                self._physics.bind(self._hole_body).pos = [0,0,0]
                self._physics.bind(self._gripper._actuator).ctrl = 0
                # self._physics.data.mocap_pos[0] = [0,0,0]
                # self._physics.data.mocap_quat[:4] = [0,0.7,0.7,0]
                # turn off gripper
                self._physics.bind(self._gripper._actuator).ctrl = 0
                self._physics.step()
                if self._render_mode == "human":
                    self._render_frame()
          

        
        print("Finish reset !!!")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # flags
        self.i = self.i + 1
        terminated = False

        # peg in hole testing logic
        if self.i < 1500: # Move peg to a position on top of peg
            peg_pos = self._physics.bind(self._peg_pickup).xpos.copy()
            peg_pos[2] = peg_pos[2]
            self._target.set_mocap_pose(self._physics, position=peg_pos[:3], quaternion=[0, 0, 0, 1])
        elif self.i < 2000: # Move peg to a position on top of peg
            peg_pos = self._physics.bind(self._peg_pickup).xpos.copy()
            peg_pos[2] = peg_pos[2] - 0.022
            self._target.set_mocap_pose(self._physics, position=peg_pos[:3], quaternion=[0, 0, 0, 1])
        elif self.i < 2500: # pickup
            self._physics.bind(self._gripper._actuator).ctrl = 1
            # self._physics.bind(self._weld).active = 1
        elif self.i < 3500: # Move above the hole
            hole_pos = self._physics.bind(self._hole).xpos.copy()
            hole_pos[2] = hole_pos[2] + 0.1
            self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0, 1])
        elif self.i < 4500: # assemble
            hole_pos = self._physics.bind(self._hole).xpos.copy()
            hole_pos[2] = hole_pos[2] + 0.002
            self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0, 1])
        elif self.i < 5500: # rotate peg inside the hole
            hole_pos = self._physics.bind(self._hole).xpos.copy()
            hole_pos[2] = hole_pos[2] + 0.002
            self._target.set_mocap_pose(self._physics, position=hole_pos[:3], quaternion=[0, 0, 0.7071068, 0.7071068])
        else:
            terminated = True

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
            self._camera._renderer.render()
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
                # cv2.imshow("hand_camera",cv2.cvtColor(self._hand_camera.image, cv2.COLOR_RGB2BGR) )
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

    def complie_model(self):
        self.close()
        # # Get Hole position
        # self._hole = self._arena.mjcf_model.find('site', "hole_site")

        # # Get Peg position
        # self._peg_pickup = self._arena.mjcf_model.find('site', "peg_pickup")
        # self._peg = self._arena.mjcf_model.find('site', "peg_base")

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # Camera 
        self._camera = Camera([480, 320], self._physics.model.ptr, self._physics.data.ptr, "fixed_camera")
        
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

        # for GUI and time keeping
        self._viewer = None
        self._timestep = self._physics.model.opt.timestep
        self._step_start = None
        self.i = 0
        print("Finish Compiling")
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
from ur3e_tasks.controllers import EEFVelocityController
from ur3e_tasks.robots import Camera
import cv2
from PIL import Image
from manipulator_mujoco.utils.transform_utils import (
    mat2quat,
)

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
        self.random_once = False
        self.random_domain = True
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

        # mocap target for viuslizing ee pose
        self._ee_pose = self._arena.mjcf_model.worldbody.add("body", name="ee_pose", mocap=True)
        self._ee_pose.add(
            "geom",
            type="box",
            size=[0.03, 0.015 ,0.015],
            rgba=[1, 0, 0, 0.2],
            conaffinity=0,
            contype=0,
            group=1
        )


        self._randomizer = DomainRandomizer(self._arena._mjcf_model, self._arm)

        # self._weld = self._gripper.setup_weld(self._arena.mjcf_model,"peg" )
        # self.complie_model()
        
        # # generate model
        # self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # # Camera 
        # self._camera1 = Camera([480, 320], self._physics.model.ptr, self._physics.data.ptr, "fixed_camera")
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
        """
        - {"frame1": np.array
        frame2": np.array
        frame3": np.array
        ee_pose: np.array
        joints: np.array
        }
        """
        if self.i % 100 == 0:
            # prepare data
            # get ee pose respect to base
            base = self._arena.mjcf_model.find('body', "ur3e/base")
            ee_pose = self.pose_in_base(self._arm.eef_site, base) # [x,y,z,qx,qy,qz,qw]
            
            # get joints 
            joints = np.array(self._physics.bind(self._arm.joints).qpos)
            # # # get images
            img1 = self.prepare_image(self._camera1.image) # [H W C]
            img2 = self.prepare_image(self._camera2.image)
            img3 = self.prepare_image(self._hand_camera.image)

            
            # # construc the dict 
            self._obs = {"frame1": img1,
            "frame2": img2,
            "frame3": img3,
            "ee_pose": ee_pose.reshape(7,1),
            "joints": joints.reshape(6,1),
            }     

        return self._obs

    def _get_info(self) -> dict:
        """
        - {"hole_pose": np.array
        "step": int
        "success": int }
        """
        # get hole pose respect to base
        base = self._arena.mjcf_model.find('body', "ur3e/base")
        hole = self._arena.mjcf_model.find('body', "hole")
        hole_pose = self.pose_in_base(hole, base)

        success = self._bt.success 

        dict = {"hole_pose":hole_pose.reshape(7,1),
                "step":self.i,
                "success":int(success)
        }   


        return dict

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        # reset flags
        self.i = 0
        # domain randomize
        if self.random_domain:
            hole_pos = self._randomizer.random_object_in_ws("hole")
            self._randomizer.random_texture("table_top")
            self._randomizer.random_texture("wall_left")
            self._randomizer.random_texture("wall_right")
            self._randomizer.random_texture("wall_back")
            self._randomizer.random_texture("wall_front")
            self._randomizer.random_light("light_source")
            self._randomizer.random_object_color("hole")
            self._randomizer.random_object_color("ur3e/assembly_ee/peg_ee")
            self._randomizer.random_object_color("ur3e/assembly_ee/peg_ee_base")
            self._randomizer.random_arm_height("ur3e/base")
            self._randomizer.random_fixed_camera("fixed_camera1","camera_center1")
            self._randomizer.random_fixed_camera("fixed_camera2","camera_center2")
            self._randomizer.random_distractors()
            self._randomizer.random_texture_arm(self._arm)
            init_pose = self._randomizer.random_initial_position(hole_pos.copy())
            self._random_state = self._randomizer.get_random_state()
            
            if self.random_once == True:
                self.random_domain = False
        else: 
            # just to setup some variable
            hole_pos = self._randomizer.random_object_in_ws("hole")
            self._randomizer.random_texture("table_top")
            self._randomizer.random_texture("wall_left")
            self._randomizer.random_texture("wall_right")
            self._randomizer.random_texture("wall_back")
            self._randomizer.random_texture("wall_front")
            self._randomizer.random_light("light_source")
            self._randomizer.random_object_color("hole")
            self._randomizer.random_object_color("ur3e/assembly_ee/peg_ee")
            self._randomizer.random_object_color("ur3e/assembly_ee/peg_ee_base")
            self._randomizer.random_arm_height("ur3e/base")
            self._randomizer.random_fixed_camera("fixed_camera1","camera_center1")
            self._randomizer.random_fixed_camera("fixed_camera2","camera_center2")
            self._randomizer.random_distractors()
            self._randomizer.random_texture_arm(self._arm)
            init_pose = self._randomizer.random_initial_position(hole_pos.copy())
            
            self._randomizer.apply_random_state(self._random_state )
            init_pose = self._random_state["initial_pose"]
        
        # recomplie model
        self.complie_model()

        # # find initial arm position
        # init_pos = self._randomizer.get_random_ws_pos_clearance(self._physics.bind(self._hole).xpos.copy(), 0.3)
        # init_quat = self._randomizer.random_quaternion_around_axis([0, 0, 0, 1],['x','y','z'], np.pi/8)
        print("Init Pose: {}".format(init_pose))
        # time.sleep(50)
        self._target.set_mocap_pose(self._physics, position=init_pose[:3], quaternion=init_pose[3:])
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
                # put arm in a reasonable starting position
                

                # random initial position
                vel_cmd = self._controller.cal_vel_from_target(target_pose)
                self._controller.run(vel_cmd)

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
        # action = [0.267633,  0., -0., -0., -0., -0.]
        print(action)

        # behavior
        # cmd, state, success, terminated = self._bt.run()
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
    
        self._controller.run(action)

        # step physics
        self._physics.step()
        # time.sleep(0.01)

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        terminated = self._bt.terminate 
            
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
            # self._camera1._renderer.render()
            print("Start viewer")
            

        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()
            

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()
            # render camera
            if self.show_cam and self.i%20 == 0:
                # Convert images to BGR format
                fixed_camera1_img = cv2.cvtColor(self._camera1.image, cv2.COLOR_RGB2BGR)
                fixed_camera2_img = cv2.cvtColor(self._camera2.image, cv2.COLOR_RGB2BGR)
                hand_camera_img = cv2.cvtColor(self._hand_camera.image, cv2.COLOR_RGB2BGR)

                # Resize images to the same size if necessary
                height = min(fixed_camera1_img.shape[0], fixed_camera2_img.shape[0], hand_camera_img.shape[0])
                width = min(fixed_camera1_img.shape[1], fixed_camera2_img.shape[1], hand_camera_img.shape[1])

                fixed_camera1_img = cv2.resize(fixed_camera1_img, (width, height))
                fixed_camera2_img = cv2.resize(fixed_camera2_img, (width, height))
                hand_camera_img = cv2.resize(hand_camera_img, (width, height))

                # Concatenate images horizontally (side by side)
                combined_image = cv2.hconcat([fixed_camera1_img, fixed_camera2_img, hand_camera_img])

                # Display all images in one window
                cv2.imshow("All Cameras", combined_image)
                cv2.waitKey(1)
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
            self._camera1._renderer.close()
            self._camera2._renderer.close()
            self._hand_camera._renderer.close()

    def complie_model(self):
        self.close()

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # Camera 
        self._camera1 = Camera([426, 240], self._physics.model.ptr, self._physics.data.ptr, "fixed_camera1")
        self._camera2 = Camera([426, 240], self._physics.model.ptr, self._physics.data.ptr, "fixed_camera2")
        self._hand_camera = Camera([426, 240],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")
        
        # self._hand_camera = Camera([400,400],self._physics.model.ptr,self._physics.data.ptr, "ur3e/hand_camera")

        # set up OSC controller
        self._controller = EEFVelocityController(
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

    def set_replay(self, replay,random_state):
        self.random_domain = False
        
        self._random_state = random_state

        # setup repaly 
        self._vel_replay = []
        # fill up with 0 from 0 step to first recored step

        for j in range(0,replay[0]["step"]):
                self._vel_replay.append(np.zeros(6))
        
        for i in range(len(replay)-1):
            
            
            for j in range(replay[i]["step"] ,replay[i+1]["step"]):
                self._vel_replay.append(replay[i]["ee_vel"].reshape(6))
            


    ##########################################################
    ### Actions
    ##########################################################
    def get_action_bt(self):
        cmd, state, success, terminated = self._bt.run()

        
        vel_cmd = np.zeros(6)
        if cmd is not None:
            self._target.set_mocap_pose(self._physics, position=cmd[0][:3], quaternion=cmd[0][3:])
            # set target for ee
            target_pose = self._target.get_mocap_pose(self._physics)

            # run vel controller to move to target pose
            vel_cmd = self._controller.cal_vel_from_target(target_pose,cmd[1],cmd[2])

        return vel_cmd,  state, success, terminated
    
    def get_action_replay(self):
        if self.i > len(self._vel_replay)-1:
            vel_cmd = np.zeros(6)
        else:
            vel_cmd = self._vel_replay[self.i]
        return vel_cmd

    ###############################################################
    #### Utils Functions 
    ###############################################################


    def pose_in_base(self,target,base):
        """
        find target pose respect to the base 
        """

        # extract pose form input 
        target_pos = self._physics.bind(target).xpos
        target_rot = self._physics.bind(target).xmat.reshape(3, 3)

        base_pos = self._physics.bind(base).xpos
        base_rot = self._physics.bind(base).xmat.reshape(3, 3)



        # calculate position
        target_pos_base = base_rot.T @ (target_pos - base_pos)

        # calculate rotation
        target_rot_base = base_rot.T @ target_rot

        # change to [x,y,z,qx,qy,qz,qw] format 
        target_quat_base = mat2quat(target_rot_base)
        target_pose = np.concatenate([target_pos_base, target_quat_base])

        # # test
        # position = base_rot @ target_pos_base + base_pos
        # mat = base_rot @ target_rot_base
        # quat = mat2quat(mat)
        # pose = np.concatenate([position,quat])
        # print("EE pos2: {}".format(pose))

        return target_pose
    
    def prepare_image(self, img):
        """
        Resize the image to render resolution, then crop it around the center to the desired size.

        Args:
            img (np.array): Input image (H, W, C)

        Returns:
            np.array: Processed image
        """

        # Define target resolutions
        render_resolution = (426, 240)  # Resize to this first
        crop_size = (224,224) # Final desired size # ideally divided by 32

        # Step 1: Resize to render resolution
        img_resized = self.resize(img, render_resolution)

        # Step 2: Crop around the center
        h, w, _ = img_resized.shape
        crop_w, crop_h = crop_size

        # Compute center
        center_x, center_y = w // 2, h // 2

        # Compute cropping box
        x1, y1 = center_x - crop_w // 2, center_y - crop_h // 2
        x2, y2 = center_x + crop_w // 2, center_y + crop_h // 2

        # Crop the image
        img_cropped = img_resized[y1:y2, x1:x2]

        return img_cropped
    
    def resize(self,im, size, im_type="rgb"):
        w, h = size
        pil_im = Image.fromarray(im)
        
        pil_im = pil_im.resize((w, h), Image.BILINEAR)
        
        im = np.asarray(pil_im)
        return im

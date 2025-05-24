import numpy as np
import os
import random
import colorsys
import copy
from scipy.spatial.transform import Rotation as R

class DomainRandomizer:
    def __init__(self, model, texture_dir = "",seed=None):
        if seed:
            np.random.seed(seed)
        self._model = model
        self.ws = self.create_ws(model)
        # self.texture_list = self.create_texture_list(texture_dir)
        self.random_objs = []

        # self.max_distractors = 15
        # self.init_random_distractor(self.max_distractors
        #                             )
        self.default_value = {}

        # for reproducing
        self.random_state = {
            "object_in_ws": {},
            "object_color": {},
            "texture": {},
            "arm_texture": {},
            "light": {},
            "height": {},
            "camera": {},
            "distractor": {}
        }
        


    ############################################################
    ## 1. Random Object Position in Workspace
    ############################################################
    def random_object_in_ws(self, object_name):
        """
        Random object inside workspace 
        """        
        if object_name+"_body" not in self.default_value:
            self.default_value[object_name+"_body"] = copy.deepcopy(self._model.find('body',object_name)._get_attribute("quat").copy())
        # random object position
        pos = self.get_random_ws_pos()
        # random object orientation
        quat = self.default_value[object_name+"_body"].copy()
        quat = self.random_quaternion_around_axis(quat, ['x'],np.pi/2,-np.pi/2 )
        quat = self.random_quaternion_around_axis(quat, ['y'],np.pi/18,0)
        quat = self.random_quaternion_around_axis(quat, ['z'],np.pi/18,-np.pi/18)

        self.apply_object_in_ws(object_name,pos,quat)

        # save the radomed value
        self.random_state["object_in_ws"][object_name] = [pos,quat]
        


    def apply_object_in_ws(self, object_name,pos,quat):
        # modify object postion in model
        self._model.find('body', object_name)._set_attribute("pos",pos)
        self._model.find('body', object_name)._set_attribute("quat",quat)

    ############################################################
    ## 2. Random Object Color
    ############################################################
    def random_object_color(self, material_name):
        """
        - randomize object colors by sampling the Hue, Saturation and Value (Brightness) (HSV)around their nominal value.
        - We sample an offset from range [−ϕo, ϕo] and add it to the object color expressed using HSV coordinates in [0, 1]
        """    
        if material_name not in self.default_value:
            self.default_value[material_name] = copy.deepcopy(self._model.find('texture',material_name)._get_attribute("rgb1").copy())

        # get object original color
        rgb = copy.deepcopy(self.default_value[material_name])

        # random rgba value around original color
        rgb = self.random_rgb(rgb,0.02)
        
        self.apply_object_color(material_name,rgb)

        # save the radomed value        
        self.random_state["object_color"][material_name] = [rgb]

    def apply_object_color(self, material_name,rgb):
         # modify object color in model
        self._model.find('texture', material_name)._set_attribute("rgb1",rgb)


    ############################################################
    ## 3. Random Texture
    ############################################################
    def random_texture(self, geom_name,root = None):
        """
        random texture of the object
        """
        if root == None:
            root = self._model

        if geom_name not in self.random_objs:
            self.create_random_assets([geom_name])
            self.random_objs.append(geom_name)

        # random texture
        texture_file = self.get_random_png()
        
        self.apply_texture(geom_name,texture_file,root)
        
        # save the radomed value        
        self.random_state["texture"][geom_name] = [texture_file,root]
        

    def random_texture_arm(self, arm):
        """
        random texture of the arm
        """
        geom_list = arm._mjcf_root.find_all("geom")
        for i, geom in enumerate(geom_list):
            geom_class = geom._get_attribute("class")
            if geom_class is not None and geom_class._get_attribute("class") == "visual":
                geom_name = geom._get_attribute("mesh")._get_attribute("name")
                geom._set_attribute("name",geom_name)
                if geom_name not in self.random_objs:
                    self.create_random_assets([geom_name], arm._mjcf_root)
                    self.random_objs.append(geom_name)

                self.random_texture(geom_name, arm._mjcf_root)
    
    def apply_texture(self, geom_name,texture_file,root):
         # modify materials
        root.find('texture', "random_{}".format(geom_name))._set_attribute("file",texture_file)

        # set material to corresponding random asset
        
        root.find('geom', geom_name)._set_attribute("material", "random_{}".format(geom_name))

        

    ############################################################
    ## 4. Random Lighthing
    ############################################################
    def random_light(self, light_name):
        """
        - the light position uniformly in a portion of a sphere for rendering images. 
        - the light source to the workspace center in the range [1.0, 3.0] meter, azimuthal angle and polar angle in the range [0, π/2] and [π/10, 4π/10] radians
        - ambient coefficients around a nominal value set to 0.3 by adding an offset sampled from the range [−ψl, ψl] where ψl is a parameter to be optimized
        """
        # random position
        light_pos = self.get_random_light_pos([1,3],[-np.pi/4,np.pi/4],[np.pi/10,4*np.pi/10])

        # rabdom ambient
        offset1 = float(np.random.uniform(-0.1,0.1))
        offset2 = float(np.random.uniform(-0.1,0.1))
        offset3 = float(np.random.uniform(-0.1,0.1))
        light_ambient = np.array([0.3,0.3,0.3]) + np.array([offset1,offset2,offset3])

        self.apply_light(light_name,light_pos,light_ambient)

        # save the radomed value        
        self.random_state["light"][light_name] = [light_pos,light_ambient]

    def apply_light(self, light_name,light_pos,light_ambient):
         # modify light
        self._model.find('light', light_name)._set_attribute("pos",light_pos)
        self._model.find('light', light_name)._set_attribute("ambient",list(light_ambient))

    ############################################################
    ## 5. Random Arm Height
    ############################################################
    def random_arm_height(self, arm_body):
        """
        Random arm height around original value
        """        
        if arm_body not in self.default_value:
            default_pos = copy.deepcopy(list(self._model.find('body',arm_body)._get_attribute("pos")))
            self.default_value[arm_body] = default_pos

        # random object position
        pos = copy.deepcopy(self.default_value[arm_body])
        # random object orientation
        ran = np.random.uniform(0,0.02)
        pos[2] = pos[2] + ran

        self.apply_height( arm_body,pos)

        # save the radomed value        
        self.random_state["height"][arm_body] = [pos]

        
    def apply_height(self, arm_body,pos):
        # modify object postion in model
        self._model.find('body', arm_body)._set_attribute("pos",pos)

    ############################################################
    ## 6. Random Camera Parameters
    ############################################################
    def random_fixed_camera(self, camera_name, center_name):
        """
        - randomize camera positions in simulation by sampling camera angles, locations and the field of view (FOV) around default values.
        - Camera always looking at workspace_center, so we can randomize workspace_center, camera position (x,y,z) around the normal configuration.

        """
        if camera_name not in self.default_value:
            cam_pos = copy.deepcopy(list(self._model.find('body',camera_name)._get_attribute("pos")))
            self.default_value[camera_name] = cam_pos
            cam_fov = self._model.find('camera',camera_name)._get_attribute("fovy")
            self.default_value[camera_name+"_fov"] = cam_fov
        
        if center_name not in self.default_value:
            center_pos = copy.deepcopy(list(self._model.find('body',center_name)._get_attribute("pos")))
            self.default_value[center_name] = center_pos

        # random center position
        center_pos = copy.deepcopy(self.default_value[center_name])
        center_pos[0] = center_pos[0] + np.random.uniform(-0.03,0.03) 
        center_pos[1] = center_pos[1] + np.random.uniform(-0.03,0.03) 
        center_pos[2] = center_pos[2] + np.random.uniform(-0.03,0.03) 

        # random camera position
        cam_pos = copy.deepcopy(self.default_value[camera_name])
        cam_pos[0] = cam_pos[0] + np.random.uniform(-0.03,0.03) 
        cam_pos[1] = cam_pos[1] + np.random.uniform(-0.03,0.03) 
        cam_pos[2] = cam_pos[2] + np.random.uniform(-0.03,0.03)

        # random fov
        cam_fov =  self.default_value[camera_name+"_fov"]
        cam_fov = cam_fov + np.random.uniform(-2,2)

        self.apply_camera(camera_name,cam_pos,center_name,center_pos,cam_fov)

        # save the radomed value        
        self.random_state["camera"][camera_name] = [cam_pos,center_name,center_pos,cam_fov]

    def apply_camera(self,camera_name,cam_pos,center_name,center_pos,cam_fov):
        # modify camera
        self._model.find('body', center_name)._set_attribute("pos",center_pos)
        self._model.find('body', camera_name)._set_attribute("pos",cam_pos)
        self._model.find('camera', camera_name)._set_attribute("fovy",cam_fov)

    ############################################################
    ## 7. Add Random Distractor 
    ############################################################
    def random_distractors(self):
        """
        - We add random primitive shapes as distractors, with random colours, positions, and sizes sampled from a uniform distribution.
        - the distractor need to be outside the workspace and not collliding with each other
        """
        pos_list = []
        # put every distractor outside
        for i in range(self.max_distractors):
            pos = [10,10,10]
            self._model.find("geom","distractor_{}".format(i))._set_attribute("pos", pos)

        # random number of distractor
        dis_num = np.random.randint(7,self.max_distractors)
        print(dis_num)
        for i in range(dis_num):
            # random type
            type_list = ["sphere", "capsule", "ellipsoid", "cylinder", "box"]
            geom_type = random.choice(type_list) 

            # random size based on geom type
            size = None
            if geom_type == "sphere":
                size = [np.random.uniform(0.04, 0.1)]  # Sphere has a single radius
            elif geom_type == "capsule":
                size = [np.random.uniform(0.04, 0.1), np.random.uniform(0.04, 0.15)]  # Capsule: [radius, half-length]
            elif geom_type == "ellipsoid":
                size = [
                    np.random.uniform(0.04, 0.1), 
                    np.random.uniform(0.04, 0.1), 
                    np.random.uniform(0.04, 0.1)
                ]  # Ellipsoid: [radius_x, radius_y, radius_z]

            elif geom_type == "cylinder": 
                size = [np.random.uniform(0.04, 0.1), np.random.uniform(0.04, 0.15)] # Cylinder: [radius, half-length]
            elif geom_type == "box":
                size = [
                    np.random.uniform(0.04, 0.1), 
                    np.random.uniform(0.04, 0.1), 
                    np.random.uniform(0.04, 0.1)
                ]  

            # random color
            rgb = [np.random.uniform(0,1) for _ in range(3)]

            # random position 
            pos = self.get_random_outside_ws_pos()
            while self.is_obj_collide(pos,pos_list,0.2):
                print("Random Distractor Posiiton!!!")
                pos = self.get_random_outside_ws_pos()
            print("Finish Random Distractor Posiiton!!!")
            pos_list.append(pos)

            # random orientation
            random_yaw = np.random.uniform(0, 2 * np.pi)
            quat = R.from_euler('z', random_yaw).as_quat()
            quat = [quat[3], quat[0], quat[1], quat[2]]

            # save the radomed value        
            self.random_state["distractor"]["{}".format(i)] = [geom_type,rgb,size,pos,quat]

            self.apply_distractor(i,geom_type,rgb,size,pos,quat)

    def apply_distractor(self,i,geom_type,rgb,size,pos,quat):
            # modify geom
            geom = self._model.find("geom","distractor_{}".format(i))
            geom._set_attribute("type", geom_type)
            geom._set_attribute("rgba", rgb+[1])
            geom._set_attribute("size", size)
            geom._set_attribute("pos", pos)
            geom._set_attribute("quat", quat)

            
    ##########################################
    ### Predefined random state
    ##########################################
    def apply_random_state(self,random_state):
        for fcn_name, states in random_state.items():
            random_fcn = getattr(self, f"apply_{fcn_name}", None)
            for object_name,values in states.items():
                random_fcn(object_name,*values)



    #########################################
    ### Helper functions
    #########################################
    def create_ws(self, model):
        inner = model.find('geom','inner_workspace')._get_attribute("size")[0]       
        outer = model.find('geom','outer_workspace')._get_attribute("size")[0]  
        min_angle = -np.pi/3 + np.pi/6
        max_angle = np.pi/3 + np.pi/6
        min_height = model.find('body','outer_workspace')._get_attribute("pos")[2] - model.find('geom','outer_workspace')._get_attribute("size")[1]
        max_height = model.find('body','outer_workspace')._get_attribute("pos")[2] + model.find('geom','outer_workspace')._get_attribute("size")[1]
        table_radius = model.find('geom','table_top')._get_attribute("size")[0]
        ws = {"inner":inner,
              "outer":outer,
              "min_angle":min_angle,
              "max_angle":max_angle,
              "min_height":min_height,
              "max_height":max_height,
              "table_radius":table_radius} 
        return ws

    def get_random_ws_pos(self):    
        
        # Unpack workspace parameters
        inner = self.ws["inner"]
        outer = self.ws["outer"]
        min_angle = self.ws["min_angle"]
        max_angle = self.ws["max_angle"]
        min_height = self.ws["min_height"]
        max_height = self.ws["max_height"]
        
        # Generate random radius between inner and outer
        r = np.random.uniform(inner, outer)
        
        # Generate random angle between min_angle and max_angle (converted to radians)
        theta = np.random.uniform(min_angle, max_angle)
        
        # Generate random height between min_height and max_height
        z = np.random.uniform(min_height, max_height)
        
        # Convert from polar to Cartesian coordinates
        x = r * np.math.cos(theta)
        y = r * np.math.sin(theta)

        pos = [x,y,z]
        return pos
    
    def get_random_ws_pos_clearance(self, object_pos, clearance):    
        default_pos = [0.2, 0.0, 1.1]
        for i in range(1000):
            ran_pos = self.get_random_ws_pos()
            distance_to_object = np.linalg.norm(np.array(ran_pos) - np.array(object_pos))
            if distance_to_object > clearance:
                return ran_pos

        return default_pos
    
    def get_random_outside_ws_pos(self):    
        
        # Unpack workspace parameters
        inner = self.ws["inner"]
        outer = self.ws["outer"]
        min_angle = self.ws["min_angle"]
        max_angle = self.ws["max_angle"]
        min_height = self.ws["min_height"]
        max_height = self.ws["max_height"]
        table_radius = self.ws["table_radius"]

        # Generate random angle outside min_angle and max_angle (converted to radians)
        if random.choice([True, False]):
            # Generate theta less than min_angle
            theta = np.random.uniform(-np.pi, min_angle/1.2)
        else:
            # Generate theta greater than max_angle
            theta = np.random.uniform(max_angle/1.2, np.pi)
       
        # Generate random radius between outer and table
        r = np.random.uniform(inner+0.1, table_radius-0.1)
        
        
        
        # Generate random height between min_height and max_height
        z = np.random.uniform(min_height, min_height+0.01)
        
        # Convert from polar to Cartesian coordinates
        x = r * np.math.cos(theta)
        y = r * np.math.sin(theta)

        pos = [x,y,z]
        return pos
    

    def random_quaternion_around_axis(self,original_quat, axis=['x', 'y', 'z'], max_angle=np.pi/8,min_angle=-np.pi/8):
        """
        Generate a random quaternion around the original quaternion, applying the rotation to specified axes.

        Parameters:
        - original_quat: The original quaternion as a list or array of size 4 [w, x, y, z].
        - axis: A list of axes to apply random rotation to. Should be a list containing 'x', 'y', 'z'.
        - max_angle: The maximum random angle (in radians) to apply for rotation (default is pi/8 radians).

        Returns:
        - A new quaternion [w, x, y, z] with random rotation applied to the specified axes.
        """
        # # change the format of quat to [x, y, z, w]
        # original_quat = [original_quat[1],original_quat[2],original_quat[3],original_quat[0]] 
        # Axis mapping to unit vectors
        axis_mapping = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1])
        }
        
        # Initialize an identity quaternion
        rotation_quat = R.from_quat([0, 0, 0, 1])  # Identity quaternion [x, y, z, w]
        
        # Loop through each specified axis
        for ax in axis:
            if ax in axis_mapping:
                # Generate a small random angle around the original axis
                angle = np.random.uniform(min_angle, max_angle)
                
                # Create a rotation quaternion for this axis
                axis_vector = axis_mapping[ax]
                axis_rotation = R.from_rotvec(angle * axis_vector)  # Rotation in axis-angle form
                
                # Multiply with the cumulative rotation quaternion
                rotation_quat = axis_rotation * rotation_quat
        
        # Combine the original quaternion with the random rotation
        original_rotation = R.from_quat(original_quat)  # Convert original quaternion to scipy rotation
        new_rotation = rotation_quat * original_rotation  # Apply random rotation
        new_rotation = new_rotation.as_quat()
        # Return the new quaternion (in [x, y, z, w] format)
        return new_rotation

    def get_random_light_pos(self, r_range, azimuth_range,polar_range):    
        
        """
        Convert spherical coordinates (r, azimuth, polar) to Cartesian coordinates [x, y, z].

        Parameters:
            r (float): Radius, distance from the origin.
            azimuth (float): Azimuth angle in radians (horizontal angle in the XY-plane).
            polar (float): Polar angle in radians (vertical angle from the Z-axis).

        Returns:
            [float, float, float]: Cartesian coordinates [x, y, z].
        """
        r = np.random.uniform(r_range[0], r_range[1])
        azimuth = np.random.uniform(azimuth_range[0], azimuth_range[1])
        polar = np.random.uniform(polar_range[0], polar_range[1])

        x = r * np.sin(polar) * np.cos(azimuth)
        y = r * np.sin(polar) * np.sin(azimuth)
        z = r * np.cos(polar) + 1
    
        return [x, y, z]
        
    def get_random_png(self):
        
        
        # Choose a random file from the list
        random_png = random.choice(self.texture_list)
        
        return random_png
    
    def create_texture_list(sefl,directory):
        texture_list = []
        
        # Walk through the directory and subdirectories
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".png"):
                    # Append the full path of the .png file to the list
                    texture_list.append(os.path.join(root, file))
        
        # Check if there are any .png files
        if not texture_list:
            return "No PNG files found."
        
        return texture_list

    def create_random_assets(self, object_list, root=None):
        """
        create random assets for the number of objects being randomized
        prevent create a new one everytime that we random texture
        """
        if root == None:
            root = self._model

        for obj in object_list:
            root.asset.add(
            "texture",
            name="random_{}".format(obj),
            type="2d",
            file=self.get_random_png(),
    
        )
        
            
            root.asset.add(
            "material",
            name="random_{}".format(obj),
            texture="random_{}".format(obj),
            texrepeat= [4,4],
        )
            
    def random_rgb(self,rgb, phi_o):
        """
        Randomizes the input RGBA color by sampling the Hue, Saturation, and Value (HSV)
        around their nominal values and applying a random offset within the range [-phi_o, phi_o].

        Parameters:
            rgb (tuple): The original color in RGBA format (R, G, B), each value in [0, 1].
            phi_o (float): The range of the offset to be applied to HSV values (in [0, 1]).

        Returns:
            tuple: The new randomized color in RGBA format (R, G, B), with each value in [0, 1].
        """

        # Extract R, G, B, and A from input
        r, g, b = rgb

        # Convert RGBA to HSV (colorsys works with RGB values in [0, 1])
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Generate random offsets for H, S, and V within the range [-phi_o, phi_o]
        delta_h = np.random.uniform(-phi_o, phi_o)
        delta_s = np.random.uniform(-phi_o, phi_o)
        delta_v = np.random.uniform(-phi_o, phi_o)

        # Apply the offsets to the original HSV values and clamp the values to valid ranges
        h = (h + delta_h) % 1.0  # Ensure Hue stays in [0, 1]
        s = np.clip(s + delta_s, 0.0, 1.0)  # Clamp Saturation to [0, 1]
        v = np.clip(v + delta_v, 0.0, 1.0)  # Clamp Value (Brightness) to [0, 1]

        # Convert the modified HSV back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        # Return the randomized color in RGBA format
        return [r, g, b]
    
    def init_random_distractor(self, dis_num):
        """
        - reserve space for distractors
        """
        distractors_list = []
        for i in range(dis_num):
            obj = self._model.worldbody.add(
            "geom",
            name="distractor_{}".format(i),
            type="box",
            size="0.2 0.2 0.2"
        )
            
    def is_obj_collide(self, pos, pos_list, clearance=0.1):
        """
        Check if the position 'pos' collides with any positions in 'pos_list' 
        given the specified 'clearance'.
        
        Parameters:
            pos (list or np.array): The position to check for collisions [x, y, z].
            pos_list (list of lists or np.array): The list of positions to check against.
            clearance (float): The minimum distance allowed between positions.
        
        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        for existing_pos in pos_list:
            # Calculate Euclidean distance between the two points in 3D
            distance = np.linalg.norm(np.array(pos) - np.array(existing_pos))
            if distance < clearance:
                return True  # Collision detected
        
        return False  # No collision
    
    def get_random_state(self):
        return self.random_state
    
    def get_random_quat(self,default):
        # random object orientation
        quat = default
        quat = self.random_quaternion_around_axis(quat, ['x'],np.pi/20,-np.pi/20)
        quat = self.random_quaternion_around_axis(quat, ['y'],np.pi/20,-np.pi/20)
        quat = self.random_quaternion_around_axis(quat, ['z'],np.pi/20,-np.pi/20)

        return quat
    
    def set_seed(self,seed):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)
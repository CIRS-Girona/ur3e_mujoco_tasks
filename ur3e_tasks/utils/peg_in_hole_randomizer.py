import numpy as np
import random
import copy
from scipy.spatial.transform import Rotation as R

class PegInHoleRandomizer:
    def __init__(self,model,seed=None):
        if seed:
            self.set_seed(seed)
        self._model = model
        self.ws = self.create_ws(model)
        self.random_objs = []
        self.default_value = {}

        
    #########################################
    ### Helper functions
    #########################################
    def create_ws(self, model):
        """
            Generates the workspace for randomization.
        """
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
        """
            Get a random position inside the workspace.
        """
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
    

    def random_quaternion_around_axis(self,original_quat, axis=['x', 'y', 'z'], max_angle=np.pi/8,min_angle=-np.pi/8): #KEEP
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
    
    def get_random_quat(self,default):
        """
            Get a randomized quaternion value around a default value.
        """
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

    ############################################################
    # Random Arm Height
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
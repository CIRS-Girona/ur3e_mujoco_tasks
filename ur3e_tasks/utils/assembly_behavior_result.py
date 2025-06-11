#!/usr/bin/env python

import py_trees
import numpy as np
import time
from manipulator_mujoco.utils.controller_utils import (
    task_space_inertia_matrix,
    pose_error,
)
from manipulator_mujoco.utils.transform_utils import (
    mat2quat,
)

from scipy.spatial.transform import Rotation as R

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

def are_close(pose1, pose2, xy_thes=0.005, z_thes=0.05, ang_thes=0.005):
    err = pose_error(pose1, pose2)

    xy_err = np.linalg.norm(err[0:2])  # x and y only
    z_err = abs(err[2])                # z error separately
    ang_err = np.linalg.norm(err[3:])  # angular error

    if xy_err < xy_thes and z_err < z_thes and ang_err < ang_thes:
        return True

    return False # If both position and orientation are close enough, return True

import numpy as np

def plus_quat(quat, angle,axis='z'):
    # Normalize the input quaternion (in case it's not normalized)
    quat = np.array(quat)
    quat = quat / np.linalg.norm(quat)

    # Calculate half of the angle for quaternion calculations
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    cos_half_angle = np.cos(half_angle)

    # Create the quaternion for the specified axis of rotation
    if axis == 'x':
        rotation_quat = np.array([sin_half_angle, 0, 0, cos_half_angle])  # Rotation around x-axis
    elif axis == 'y':
        rotation_quat = np.array([0, sin_half_angle, 0, cos_half_angle])  # Rotation around y-axis
    elif axis == 'z':
        rotation_quat = np.array([0, 0, sin_half_angle, cos_half_angle])  # Rotation around z-axis
    else:
        raise ValueError("Invalid axis, choose from 'x', 'y', or 'z'.")

    # Multiply the quaternions (quat * rotation_quat)
    x1, y1, z1, w1 = quat
    x2, y2, z2, w2 = rotation_quat

    new_quat = [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ]

    return new_quat


def plus_pose(pose, distance, axis='x'):
    # Extract position and quaternion from pose
    position = np.array(pose[:3])  # [x, y, z]
    quat = pose[3:]  # [qx, qy, qz, qw]

    # Convert quaternion to a rotation matrix
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()  # 3x3 rotation matrix

    # Define the translation vector based on the axis
    if axis == 'x':
        translation = np.array([distance, 0, 0])  # Translation along local x-axis
    elif axis == 'y':
        translation = np.array([0, distance, 0])  # Translation along local y-axis
    elif axis == 'z':
        translation = np.array([0, 0, distance])  # Translation along local z-axis
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # Apply the translation in the local frame (considering the rotation)
    translated_position = position + rotation_matrix @ translation

    # The orientation (quaternion) remains the same
    new_pose = np.concatenate([translated_position, quat])

    return new_pose



class SetUp(py_trees.behaviour.Behaviour):
    def __init__(self, name, physics,eef,hole, update_period):
        super(SetUp, self).__init__(name)
        
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key("physics", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("eef", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("hole", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("command", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("state_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("success", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("terminate", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("update_period", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("dagger", access=py_trees.common.Access.WRITE)



        self.blackboard.physics = physics
        self.blackboard.eef = eef
        self.blackboard.hole = hole

        self.blackboard.command = None
        self.blackboard.success = False
        self.blackboard.terminate = False
        self.blackboard.update_period = update_period
        self.blackboard.dagger = True
        

        

    def setup(self):
        self.logger.debug("  %s [SetUp::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [SetUp::initialise()]" % self.name)
        self.counter = 0
        self.blackboard.state_result = 0


    def update(self):
        return py_trees.common.Status.SUCCESS
        

    def terminate(self, new_status):
        self.logger.debug("  %s [SetUp::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))




class MoveToHole(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(MoveToHole, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key("physics", access=py_trees.common.Access.READ)
        self.blackboard.register_key("eef", access=py_trees.common.Access.READ)
        self.blackboard.register_key("hole", access=py_trees.common.Access.READ)
        self.blackboard.register_key("update_period", access=py_trees.common.Access.READ)
        self.blackboard.register_key("command", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("state_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("success", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("terminate", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("dagger", access=py_trees.common.Access.READ)

        self.physics = self.blackboard.physics
        self.eef = self.blackboard.eef
        self.hole = self.blackboard.hole

        self.vel_lim = 0.1
        self.ang_vel_lim = 0.35

        self.offset = 0.1
        self.max_counter = 10000/self.blackboard.update_period
        self.max_counter = 999999999

        self.close_counter = 0
        self.close_threshold = 30 # 50

    def setup(self):
        self.logger.debug("  %s [MoveToHole::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [MoveToHole::initialise()]" % self.name)
        self.blackboard.state_result = 0
        self.counter = 0
        hole_pos = self.physics.bind(self.hole).xpos.copy()
        hole_quat = mat2quat(self.physics.bind(self.hole).xmat.reshape(3, 3))
        self.hole_pose = np.concatenate([hole_pos, hole_quat])
        self.hole_pose = plus_pose(self.hole_pose,self.offset,axis='z')
        # self.hole_pose = plus_pose(self.hole_pose,0.05,axis='x')

    def update(self):
        self.counter = self.counter + 1
        self.blackboard.success = False
        self.blackboard.terminate = False
        if self.counter > self.max_counter:
            self.logger.debug(f"  {self.name}: Timeout exceeded")
            self.blackboard.success = False
            self.blackboard.terminate = True
            return py_trees.common.Status.FAILURE
        
        

        eef_pose = self.physics.bind(self.eef).xpos
        eef_quat = mat2quat(self.physics.bind(self.eef).xmat.reshape(3, 3))
        eef_pose = np.concatenate([eef_pose, eef_quat])

        if are_close(self.hole_pose,eef_pose,xy_thes=0.035,z_thes=0.1,ang_thes=0.3) :
            if self.close_counter >=self.close_threshold:
                self.close_counter = 0
                self.logger.debug("MoveToHole SUCCESS!!!")
                return py_trees.common.Status.SUCCESS
            else:
                self.close_counter = self.close_counter +1
                self.logger.debug("MoveToHole Alomost SUCCESS!!!")
                return py_trees.common.Status.RUNNING
        else:
            self.blackboard.command = [self.hole_pose,self.vel_lim,self.ang_vel_lim]
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [MoveToHole::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
class Assemble(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Assemble, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key("physics", access=py_trees.common.Access.READ)
        self.blackboard.register_key("eef", access=py_trees.common.Access.READ)
        self.blackboard.register_key("hole", access=py_trees.common.Access.READ)
        self.blackboard.register_key("update_period", access=py_trees.common.Access.READ)

        self.blackboard.register_key("command", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("state_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("success", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("terminate", access=py_trees.common.Access.WRITE)

        self.physics = self.blackboard.physics
        self.eef = self.blackboard.eef
        self.hole = self.blackboard.hole

        self.vel_lim = 0.035 #0.035
        self.ang_vel_lim = 0.2

        self.offset = 0.0
        self.max_counter = 8000/self.blackboard.update_period
        self.max_counter = 999999999

    def setup(self):
        self.logger.debug("  %s [Assemble::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Assemble::initialise()]" % self.name)
        self.blackboard.state_result = 1
        self.counter = 0
        hole_pos = self.physics.bind(self.hole).xpos.copy()
        hole_pos[2] = hole_pos[2] + self.offset
        hole_quat = mat2quat(self.physics.bind(self.hole).xmat.reshape(3, 3)) # [x,y,z,w]
        self.hole_pose = np.concatenate([hole_pos, hole_quat])
    def update(self):
        self.counter = self.counter + 1
        self.blackboard.success = False
        self.blackboard.terminate = False
        if self.counter > self.max_counter:
            self.logger.debug(f"  {self.name}: Timeout exceeded")
            self.blackboard.success = False
            self.blackboard.terminate = True
            return py_trees.common.Status.RUNNING
        
       

        eef_pos = self.physics.bind(self.eef).xpos.copy()
        eef_quat = mat2quat(self.physics.bind(self.eef).xmat.reshape(3, 3))
        eef_pose = np.concatenate([eef_pos, eef_quat])

        if are_close(self.hole_pose,eef_pose,z_thes=0.043,xy_thes=0.05,ang_thes=0.1) :
            self.blackboard.state_result = 2

        if are_close(self.hole_pose,eef_pose,z_thes=0.012,xy_thes=0.03,ang_thes=0.1) :
            self.logger.debug("Assemble SUCCESS!!!")
            return py_trees.common.Status.SUCCESS
        elif not are_close(self.hole_pose,eef_pose,z_thes=0.2,xy_thes=0.011,ang_thes=0.017) :
            if abs(self.hole_pose[2] - eef_pose[2]) < 0.03:
                self.blackboard.command = [self.hole_pose,self.vel_lim,self.ang_vel_lim]
            self.logger.debug("Assemble Misalign!!!")
            self.alingned_pose = self.hole_pose.copy()
            self.alingned_pose[2] = eef_pose[2] + 0.01
            self.blackboard.command = [self.alingned_pose,self.vel_lim,self.ang_vel_lim]
            return py_trees.common.Status.RUNNING
        else:
            self.blackboard.command = [self.hole_pose,self.vel_lim,self.ang_vel_lim]
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [Assemble::terminate().terminate()][%s->%s]" %
                        (self.name, self.status, new_status))
        
class Rotate(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Rotate, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key("physics", access=py_trees.common.Access.READ)
        self.blackboard.register_key("eef", access=py_trees.common.Access.READ)
        self.blackboard.register_key("hole", access=py_trees.common.Access.READ)
        self.blackboard.register_key("update_period", access=py_trees.common.Access.READ)

        self.blackboard.register_key("command", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("state_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("success", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("terminate", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("dagger", access=py_trees.common.Access.READ)

        self.physics = self.blackboard.physics
        self.eef = self.blackboard.eef
        self.hole = self.blackboard.hole

        self.vel_lim = 0.2
        self.ang_vel_lim = 1.0

        self.offset = 0.0
        self.max_counter = 4000/self.blackboard.update_period
        self.max_counter = 999999999

    def setup(self):
        self.logger.debug("  %s [Rotate::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Rotate::initialise()]" % self.name)
        self.blackboard.state_result = 3
        self.counter = 0

        eef_pos = self.physics.bind(self.eef).xpos.copy()
        eef_quat = mat2quat(self.physics.bind(self.eef).xmat.reshape(3, 3)).copy()
        eef_pose = np.concatenate([eef_pos, eef_quat])

        self.rotated_pose = eef_pose.copy()
        self.rotated_pose[3:] = plus_quat(eef_quat.copy(), np.pi/2)

    def update(self):
        self.counter = self.counter + 1
        if self.counter > self.max_counter:
            self.logger.debug(f"  {self.name}: Timeout exceeded")
            self.blackboard.success = False
            self.blackboard.terminate = True
            return py_trees.common.Status.FAILURE
        
        eef_pos = self.physics.bind(self.eef).xpos.copy()
        eef_quat = mat2quat(self.physics.bind(self.eef).xmat.reshape(3, 3)).copy()
        eef_pose = np.concatenate([eef_pos, eef_quat])


        if are_close(self.rotated_pose,eef_pose,ang_thes=0.3,xy_thes=0.05,z_thes=0.05) :
            self.logger.debug("Rotate SUCCESS!!!")
            self.blackboard.success = True
            self.blackboard.terminate = True
            self.blackboard.state_result = 4
            if self.blackboard.dagger:
                self.blackboard.command = [self.rotated_pose,self.vel_lim,self.ang_vel_lim]
                return py_trees.common.Status.RUNNING
            return py_trees.common.Status.SUCCESS
        else:
            self.blackboard.command = [self.rotated_pose,self.vel_lim,self.ang_vel_lim]
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [Rotate::terminate().terminate()][%s->%s]" %
                        (self.name, self.status, new_status))




class AssemblyBTResult:
    """
    Behavior Tree class for assembling behavior
    - Manage the current goal and state for the controller 
    """

    def __init__(self, physics, eef, hole):
        # py_trees.logging.level = py_trees.logging.Level.DEBUG        
        self.counter = 0 # 
        self.update_period = 1
        # self.state = 0
        self.success = False
        self.terminate = False
        # Create Behaviors
        set_up = SetUp("set_up",physics, eef, hole,self.update_period)
        move_to_hole = MoveToHole("move_to_hole")
        assemble = Assemble("assemble")
        rotate = Rotate("rotate")
        # Sub-sequence for move and assemble
        # move_and_assemble = py_trees.composites.Sequence(name="move_and_assemble", memory=True)
        # move_and_assemble.add_children([move_to_hole, assemble])

        # # Retry decorator around the move-and-assemble sequence
        # retry_move_and_assemble = py_trees.decorators.Retry(
        #     name="Retry_Move_And_Assemble",
        #     child=move_and_assemble,
        #     num_failures=999  # adjust as needed
        # )

        # Top-level sequence
        assembly_seq = py_trees.composites.Sequence(name="assembly_seq", memory=True)
        assembly_seq.add_children([set_up, move_to_hole, assemble, rotate])
        

        self.bt = assembly_seq
        py_trees.display.render_dot_tree(self.bt )
        print("Call setup for all tree children")
        self.bt .setup_with_descendants() # call setup() of all behaviors in the tree (set up ROS topic, service ...)
        print("Setup done!\n\n")
        py_trees.display.ascii_tree(self.bt )
        
    
    def run(self):
        self.counter += 1
        self.command = self.bt.children[0].blackboard.command
        # self.state = self.bt.children[0].blackboard.state_result
        self.success = self.bt.children[0].blackboard.success
        self.terminate = self.bt.children[0].blackboard.terminate
        if self.counter % self.update_period == 0: 
            self.bt.tick_once()
        print("state: ", self.bt.children[0].blackboard.state_result)
        return self.command, self.bt.children[0].blackboard.state_result, self.success, self.terminate # 

    def reset(self):
        pass

    


    # root = py_trees.composites.Parallel(name="root",policy= py_trees.common.ParallelPolicy.SuccessOnSelected([is_finished]))
    # root.add_children([is_finished,repeat_pick_place])
    
    
    
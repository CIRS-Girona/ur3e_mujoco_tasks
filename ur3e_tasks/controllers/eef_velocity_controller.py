from manipulator_mujoco.controllers import JointEffortController

import numpy as np

from manipulator_mujoco.utils.controller_utils import (
    task_space_inertia_matrix,
    pose_error,
)

from manipulator_mujoco.utils.mujoco_utils import (
    get_site_jac, 
    get_fullM
)

from manipulator_mujoco.utils.transform_utils import (
    mat2quat,
)

class EEFVelocityController(JointEffortController):
    def __init__(
        self,
        physics,
        joints,
        eef_site,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
        kv: float,
        kp: float=0.0,
        ko: float=0.0,
        vmax_xyz: float=0.0,
        vmax_abg: float=0.0,
    ) -> None:
        
        super().__init__(physics, joints, min_effort, max_effort)

        self._eef_site = eef_site
        self._kv = kv
        self._kp = kp
        self._ko = ko
        self._vmax_xyz = vmax_xyz
        self._vmax_abg = vmax_abg

        self._eef_id = self._physics.bind(eef_site).element_id
        self._jnt_dof_ids = self._physics.bind(joints).dofadr
        self._dof = len(self._jnt_dof_ids)

        self._task_space_gains = np.array([self._kp] * 3 + [self._ko] * 3)
        self._lamb = self._task_space_gains / self._kv
        # self._sat_gain_xyz = vmax_xyz / self._kp * self._kv
        # self._sat_gain_abg = vmax_abg / self._ko * self._kv
        # self._scale_xyz = vmax_xyz / self._kp * self._kv
        # self._scale_abg = vmax_abg / self._ko * self._kv

    def run(self, target_vel):
        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(
            self._physics.model.ptr, 
            self._physics.data.ptr, 
            self._eef_id,
        )
        J = J[:, self._jnt_dof_ids]

        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(
            self._physics.model.ptr, 
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self._physics.bind(self._joints).qvel

        # Compute current end-effector velocity (J @ dq)
        v_current = J @ dq

        # Velocity error (task-space)
        v_error = target_vel - v_current

        # joint space control signal
        u = np.zeros(self._dof)
        
        ## Feedforward term: J^T * Mx * v_desired (optional, for better tracking)
        # Add the task space control signal to the joint space control signal
        u += np.dot(J.T, np.dot(Mx, target_vel))

        # Add damping to joint space control signal
        # u += self._kv * np.dot(M, v_error)

        ## Feedback term: Damping to stabilize velocity error
        u += np.dot(J.T, self._kv * v_error)  # kv is now a velocity damping gain

        # Add gravity compensation to the target effort
        u += self._physics.bind(self._joints).qfrc_bias

        # send the target effort to the joint effort controller
        super().run(u)

    def cal_vel_from_target(self, target, xyz_lim = 1, abg_lim = 2):
        # target pose is a 7D vector [x, y, z, qx, qy, qz, qw]
        target_pose = target
         # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        ee_pos = self._physics.bind(self._eef_site).xpos
        ee_quat = mat2quat(self._physics.bind(self._eef_site).xmat.reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])
        # print("EE Pose1: {}".format(ee_pose))

        # Calculate the pose error (difference between the target and current pose).
        pose_err = pose_error(target_pose, ee_pose)
        
        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)

        # Calculate the task space control signal.
        u_task += self.compute_ee_velocity(pose_err,2,3,xyz_lim,abg_lim)

        return u_task
    

    def compute_ee_velocity(self,pose_error: np.ndarray, 
                        Kp: float = 1.0, 
                        Ko: float = 3.0,
                        max_linear_speed: float = 0.5, 
                        max_angular_speed: float = 1.0) -> np.ndarray:
        """
        Compute the end-effector velocity based on pose error, with velocity limits.

        Parameters:
            pose_error (np.ndarray): Pose error (6D vector) [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
            Kp (float): Proportional gain for control (default = 1.0)
            max_linear_speed (float): Maximum linear velocity (default = 0.5 m/s)
            max_angular_speed (float): Maximum angular velocity (default = 1.0 rad/s)

        Returns:
            np.ndarray: EE velocity [vx, vy, vz, wx, wy, wz] with limits applied
        """
        # Compute raw velocity using proportional control
        linear_velocity = Kp * pose_error[:3]  # Simple P-controller
        angular_velocity = Ko * pose_error[3:]

        # Compute norms
        linear_norm = np.linalg.norm(linear_velocity)
        angular_norm = np.linalg.norm(angular_velocity)

        # Apply speed limits
        if linear_norm > max_linear_speed:
            linear_velocity = (linear_velocity / linear_norm) * max_linear_speed  # Scale down
        
        if angular_norm > max_angular_speed:
            angular_velocity = (angular_velocity / angular_norm) * max_angular_speed  # Scale down

        # Return the final scaled velocity
        return np.hstack([linear_velocity, angular_velocity])


    def _scale_signal_vel_limited(self, u_task: np.ndarray, xyz_lim = 1,abg_lim = 2) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > xyz_lim:
            _scale_xyz = xyz_lim / self._kp * self._kv
            scale[:3] *= _scale_xyz / norm_xyz
        if norm_abg > abg_lim:
            _scale_abg = abg_lim / self._ko * self._kv
            scale[3:] *= _scale_abg / norm_abg
        
        return self._kv * scale * self._lamb * u_task


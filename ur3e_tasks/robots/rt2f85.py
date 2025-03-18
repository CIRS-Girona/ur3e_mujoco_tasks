import os
from manipulator_mujoco.robots.gripper import Gripper

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'assets','robotiq_2f85_v4', 'mjx_2f85.xml')
xml_path = os.path.abspath(file_path)

_2F85_XML = xml_path
_JOINT = 'right_driver_joint'

_ACTUATOR = 'fingers_actuator'

class RT2F85(Gripper):
    def __init__(self, name: str = None):
        super().__init__(_2F85_XML, _JOINT, _ACTUATOR, name)
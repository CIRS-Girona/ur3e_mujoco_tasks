import os
from manipulator_mujoco.robots.gripper import Gripper

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'assets','suction', 'suction.xml')
xml_path = os.path.abspath(file_path)

_2F85_XML = xml_path
_JOINT = 'suction_joint'

_ACTUATOR = 'suction_actuator'

class Suction(Gripper):
    def __init__(self, name: str = None):
        super().__init__(_2F85_XML, _JOINT, _ACTUATOR, name)
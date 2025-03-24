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

    
    def setup_weld(self, model,object):
        model.equality.add("connect",name="suction_weld",site1="floor_stie",site2="peg_pickup",active="false")
        weld = model.find("equality", "suction_weld")
        
        # weld.site1 = "ur3e/suction/suction_site"
        # weld.site2 = "peg_pickup"
        
        return  model.find("equality", "suction_weld")
        
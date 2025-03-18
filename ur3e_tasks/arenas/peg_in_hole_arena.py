from dm_control import mjcf
import os

class PegInHoleArena(object):
    def __init__(self) -> None:
        """
        Initializes the StandardArena object by creating a new MJCF model and adding a checkerboard floor and lights.
        """

        xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/peg-in-hole/peg_in_hole_stl.xml',
            )

        self._mjcf_model = mjcf.from_path(xml_path)

        # addlight sources
        for x in [-2, 2]:
            self._mjcf_model.worldbody.add("light", pos=[x, -1, 3], dir=[-x, 1, -2])
    
        
    
    def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model at a specified position and orientation.

        Args:
            child: The child element to attach.
            pos: The position of the child element.
            quat: The orientation of the child element.

        Returns:
            The frame of the attached child element.
        """
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame
    
    def attach_free(self, child,  pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model with a free joint.

        Args:
            child: The child element to attach.

        Returns:
            The frame of the attached child element.
        """
        frame = self.attach(child)
        frame.add('freejoint')
        frame.pos = pos
        frame.quat = quat
        return frame
    
    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """
        Returns the MJCF model for the StandardArena object.

        Returns:
            The MJCF model.
        """
        return self._mjcf_model
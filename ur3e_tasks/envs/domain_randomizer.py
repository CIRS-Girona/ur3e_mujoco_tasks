class DomainRandomizer:
    def __init__(self, model):
        self._model = model
        
        

    def random_peg_hole(self, peg_name, hole_name):
        """
        Random peg and hole in workspace
        - Hole got randomed first, with yaw distributed around 0
        - Random peg with yaw distributed around 0, then check the proximity to the hole
        """
        hole = self._model.find('body', hole_name)
        peg = self._model.find('body', peg_name)
        
        # random hole position
        self.random_object_workspace(hole)

        # random peg position
        while True:
            self.random_object_workspace(hole)


    #########################################
    ### Helper functions
    #########################################
    def get_random_ws_pose(self, object):
        

    def is_object_collided(self,obj1,obj2)
        :
        
        
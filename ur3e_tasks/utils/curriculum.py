import numpy as np

class CurriculumLearning:
    def __init__(self) -> None:
        self.final_stage = 8

    def get_target_point(self,hole_pos,hole_rot,learning_stage):
        # set intermediate point: a point above the hole
        intermediate_target_pos = hole_pos
        if learning_stage < self.final_stage:
            if learning_stage <= 3:
                offset = np.array([0,0,0.07])
            elif learning_stage == 4:
                offset = np.array([0,0,0.055])
            elif learning_stage == 5:
                offset = np.array([0,0,0.045])
            elif learning_stage == 6:
                offset = np.array([0,0,0.03])
            else: # later stages only affect force penalty
                offset = np.zeros(3)
            intermediate_target_pos += hole_rot @ offset.T

        return intermediate_target_pos
    
    def get_distance_threshold(self,learning_stage):
        # set distance threshold
        if learning_stage == 1:
            dist_threshold = 0.09
        elif learning_stage == 2:
            dist_threshold = 0.05
        else:
            dist_threshold = 0.01

        return dist_threshold
    
    def get_force_penalty(self,learning_stage):
        if learning_stage == self.final_stage:
            return -10.0
        else:
            return 0.0
    
    def check_task_completed(self,peg_pos,peg_rot,hole_pos,hole_rot,learning_stage):
        intermediate_pt = self.get_target_point(hole_pos,hole_rot,learning_stage)
        dist_threshold = self.get_distance_threshold(learning_stage)

        # reproduce transformation matrix of peg (w.r.t. world)
        peg_end_transform = np.block([[peg_rot,peg_pos.reshape(-1,1)],[0,0,0,1]])             

        # compute transformation from intermediate point to peg
        intermediate_pt_transform = np.block([[hole_rot,intermediate_pt.reshape(-1,1)],[0,0,0,1]]) 
        intermediate_pt_to_peg_transform = np.linalg.inv(intermediate_pt_transform) @ peg_end_transform 
        
        peg_wrt_intermediate_pt_pos = intermediate_pt_to_peg_transform[:3,3]
        distance_to_intermediate_pt = np.linalg.norm(peg_wrt_intermediate_pt_pos) 
                
        if learning_stage <= 2:
            # Define success = peg reaches within a semisphere with a certain radius of the intermediate point 
            task_completed = (distance_to_intermediate_pt < dist_threshold) and (peg_wrt_intermediate_pt_pos[-1] > -0.03) 
        else:
            # Define success = peg reaches the intermediate point with a very small radius
            task_completed = (distance_to_intermediate_pt < dist_threshold)

        return task_completed
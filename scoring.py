import numpy as np
import cv2
from angle_calculation import PoseAngle

class ScoringSystem:
    def __init__(self, standard_file, line_index: int) -> None:
        self.angle_list = standard_file
        self.line_index = line_index

        self.standard_list = self.get_standard_list(standard_file, line_index)
        self.demo_left_elbow_angle = float(self.standard_list[0])  # Left elbow angle
        self.demo_right_elbow_angle = float(self.standard_list[1])
        self.demo_left_shoulder_angle = float(self.standard_list[2])
        self.demo_right_shoulder_angle = float(self.standard_list[3])
        self.demo_left_hip_angle = float(self.standard_list[4])
        self.demo_right_hip_angle = float(self.standard_list[5])
        self.demo_left_knee_angle = float(self.standard_list[6])
        self.demo_right_knee_angle = float(self.standard_list[7])



    def get_standard_list(self, standard_file, line_index):
        """
        This function reads one line from the standard_file and store the angles in a list
        """
        standard_list = standard_file[line_index]
        return standard_list

    def judge(self, angle, player_image, joint1, joint2, player_frame_shape, standard) -> int:
        height, width, _ = player_frame_shape
        point = 0
        if joint2 == [0.0, 0.0] or joint1 == [0.0, 0.0]:
            return point
        
        # Convert normalized landmarks to pixel coordinates
        joint1_pixel = tuple(np.multiply(joint1, [width, height]).astype(int))
        joint2_pixel = tuple(np.multiply(joint2, [width, height]).astype(int))
        
        # # Calculate color based on angle
        # unit = 255 // 50
        # colour_offset = abs(angle - standard) * unit
        # R = min(255, colour_offset)
        # G = max(0, 255 - colour_offset)
        # line_colour = (0, G, R)  # has to be a tuple

        # # Draw the line between elbow and wrist
        

        difference = abs(angle - standard)

        if difference < 30:
            point = 3
            line_colour = (0, 255, 0) # (B, G, R)
        elif difference < 60:
            point = 2
            line_colour = (0, 255, 255)
        elif difference < 90:
            point = 1
            line_colour = (0, 140, 255)
        else:
            point = 0
            line_colour = (0, 0, 255)

        cv2.line(player_image, joint1_pixel, joint2_pixel, line_colour, 20)

        return point
    
    def get_frame_score(self, player, player_image, player_frame_shape) -> int:
        sum = 0
        sum += self.judge(player.left_elbow_angle, player_image,
                          player.left_elbow, player.left_wrist,
                          player_frame_shape, self.demo_left_elbow_angle)
        sum += self.judge(player.right_elbow_angle, player_image,
                          player.right_elbow, player.right_wrist,
                          player_frame_shape, self.demo_right_elbow_angle)
        sum += self.judge(player.left_shoulder_angle, player_image,
                          player.left_shoulder, player.left_elbow,
                          player_frame_shape, self.demo_left_shoulder_angle)
        sum += self.judge(player.right_shoulder_angle, player_image, 
                          player.right_shoulder, player.right_elbow, 
                          player_frame_shape, self.demo_right_shoulder_angle)
        sum += self.judge(player.left_hip_angle, player_image, 
                          player.left_hip, player.left_knee, 
                          player_frame_shape, self.demo_left_hip_angle)
        sum += self.judge(player.right_hip_angle, player_image, 
                          player.right_hip, player.right_knee, 
                          player_frame_shape, self.demo_right_hip_angle)
        sum += self.judge(player.left_knee_angle, player_image, 
                          player.left_knee, player.left_ankle, 
                          player_frame_shape, self.demo_left_knee_angle)
        sum += self.judge(player.right_knee_angle, player_image, 
                          player.right_knee, player.right_ankle, 
                          player_frame_shape, self.demo_right_knee_angle)
        
        return sum
        # ok but not all 8 joints will be shown all the time. What abt that
        # visability
        # total score is easily accessible, but I need to NOT compare the angle if it's "invisible",
        # lable it in some way?
    
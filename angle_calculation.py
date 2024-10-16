import cv2
import numpy as np
from enum import Enum

# Define an Enum for pose landmarks
class PoseLandmark(Enum):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

class PoseAngle:

    def __init__(self, image, frame_shape, landmarks: list) -> None:
        self.landmarks = landmarks
        self.image = image
        self.frame_shape = frame_shape
        self.visibility_threshold = 0.05
        
        # Left lankmark coordinates 
        self.left_shoulder = self.get_landmark(PoseLandmark.LEFT_SHOULDER, self.visibility_threshold)
        self.left_elbow = self.get_landmark(PoseLandmark.LEFT_ELBOW, self.visibility_threshold)
        self.left_wrist = self.get_landmark(PoseLandmark.LEFT_WRIST, self.visibility_threshold)
        self.left_hip = self.get_landmark(PoseLandmark.LEFT_HIP, self.visibility_threshold)
        self.left_knee = self.get_landmark(PoseLandmark.LEFT_KNEE, self.visibility_threshold)
        self.left_ankle = self.get_landmark(PoseLandmark.LEFT_ANKLE, self.visibility_threshold)

        # Right landmark coordinates
        self.right_shoulder = self.get_landmark(PoseLandmark.RIGHT_SHOULDER, self.visibility_threshold)
        self.right_elbow = self.get_landmark(PoseLandmark.RIGHT_ELBOW, self.visibility_threshold)
        self.right_wrist = self.get_landmark(PoseLandmark.RIGHT_WRIST, self.visibility_threshold)
        self.right_hip = self.get_landmark(PoseLandmark.RIGHT_HIP, self.visibility_threshold)
        self.right_knee = self.get_landmark(PoseLandmark.RIGHT_KNEE, self.visibility_threshold)
        self.right_ankle = self.get_landmark(PoseLandmark.RIGHT_ANKLE, self.visibility_threshold)

        # Left joint angles
        self.left_elbow_angle = self.calculate_angle(self.left_shoulder, self.left_elbow, self.left_wrist)
        self.left_shoulder_angle = self.calculate_angle(self.left_hip, self.left_shoulder, self.left_elbow)
        self.left_hip_angle = self.calculate_angle(self.left_knee, self.left_hip, self.left_shoulder)
        self.left_knee_angle = self.calculate_angle(self.left_ankle, self.left_knee, self.left_hip)

        # Right joint angles
        self.right_elbow_angle = self.calculate_angle(self.right_shoulder, self.right_elbow, self.right_wrist)
        self.right_shoulder_angle = self.calculate_angle(self.right_hip, self.right_shoulder, self.right_elbow)
        self.right_hip_angle = self.calculate_angle(self.right_knee, self.right_hip, self.right_shoulder)
        self.right_knee_angle = self.calculate_angle(self.right_ankle, self.right_knee, self.right_hip)

        # Left joint angles display
        self.display_angle(self.image, self.left_elbow_angle, self.left_elbow, self.frame_shape)
        self.display_angle(self.image, self.left_shoulder_angle, self.left_shoulder, self.frame_shape)
        self.display_angle(self.image, self.left_hip_angle, self.left_hip, self.frame_shape)
        self.display_angle(self.image, self.left_knee_angle, self.left_knee, self.frame_shape)
        
        # Right joint angles display
        self.display_angle(self.image, self.right_elbow_angle, self.right_elbow, self.frame_shape)
        self.display_angle(self.image, self.right_shoulder_angle, self.right_shoulder, self.frame_shape)
        self.display_angle(self.image, self.right_hip_angle, self.right_hip, self.frame_shape)
        self.display_angle(self.image, self.right_knee_angle, self.right_knee, self.frame_shape)

        self.visible_angle_num = self.get_angle_num()
       
    def get_landmark(self, landmark_enum: PoseLandmark, threshold) -> list:
        """
        A helper function to get the x, y coordinates of a specific landmark 
        using the Enum to reference the index in the landmarks list.
        """
        
        # check visibility, if < threshold, set joint coordinate to 0
        if self.landmarks[landmark_enum.value].visibility < threshold:
            x = 0.0
            y = 0.0

        else:
            x = self.landmarks[landmark_enum.value].x
            y = self.landmarks[landmark_enum.value].y

        return [x, y]

    def calculate_angle(self, vector_a, vector_b, vector_c) -> float:
        """
        A function to calculate the angle by the two lines connecting vector a, b and vector b, c
        """
        a = np.array(vector_a)
        b = np.array(vector_b)
        c = np.array(vector_c)

        # if any of the joints is undetectable (set to 0 by the get_landmark function), disable the angle
        if np.array_equal(a, np.zeros(2)) or np.array_equal(b, np.zeros(2)) or np.array_equal(c, np.zeros(2)):
            angle = 0.0
        else:
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
        return angle
    
    def display_angle(self, image, angle, joint, frame_shape) -> None:
        """
        A function to display the angle of the joint at the location of the joint with 20px of offset
        """
        text = str(int(angle))
        height, width, _ = frame_shape
        position =  tuple(np.multiply(joint, [width, height]).astype(int))

        # 
        position = (position[0] + 20, position[1] - 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        colour = (255, 255, 255)
        thickness = 2
        
        cv2.putText(image, text, position, font, font_scale, colour, thickness, cv2.LINE_AA)

    def get_angle_num(self) -> int:
        """
        This function spit out the number of calculatable angles by checking the visibility of the 12 angles.
        If the visibility of an angle is less then 0.70, it is considered invisible and the angle that involves it is uncalculatabe
        """
        visible_joint_num = 0
        visible_angle_num = 8
        for enum in PoseLandmark:
            if self.landmarks[enum.value].visibility > self.visibility_threshold:
                visible_joint_num += 1

            else:
                # if invisible, the four joints at the endpoints disables only one angle
                if enum in {PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST, 
                            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE}:
                    visible_angle_num -= 1

                # all other joints would affect two angles
                else:
                    visible_angle_num -= 2
        
        return visible_angle_num

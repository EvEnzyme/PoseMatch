import cv2
import numpy as np

class PoseAngle:

    def __init__(self, image, frame_shape, landmarks: list) -> None:
        self.landmarks = landmarks
        self.image = image
        self.frame_shape = frame_shape
        
        # Left lankmark coordinates 
        self.left_shoulder = [self.landmarks[11].x, self.landmarks[11].y]
        self.left_elbow = [self.landmarks[13].x, self.landmarks[13].y]
        self.left_wrist = [self.landmarks[15].x, self.landmarks[15].y]
        self.left_hip = [self.landmarks[23].x, self.landmarks[23].y]
        self.left_knee = [self.landmarks[25].x, self.landmarks[25].y]
        self.left_ankle = [self.landmarks[27].x, self.landmarks[27].y]

        # Right landmark coordinates
        self.right_shoulder = [self.landmarks[12].x, self.landmarks[12].y]
        self.right_elbow = [self.landmarks[14].x, self.landmarks[14].y]
        self.right_wrist = [self.landmarks[16].x, self.landmarks[16].y]
        self.right_hip = [self.landmarks[24].x, self.landmarks[24].y]
        self.right_knee = [self.landmarks[26].x, self.landmarks[26].y]
        self.right_ankle = [self.landmarks[28].x, self.landmarks[28].y]
      
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
       
    def calculate_angle(self, vector_a, vector_b, vector_c) -> float:
        a = np.array(vector_a)
        b = np.array(vector_b)
        c = np.array(vector_c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def display_angle(self, image, angle, joint, frame_shape):
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


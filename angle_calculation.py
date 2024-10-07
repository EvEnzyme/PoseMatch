import cv2
import numpy as np

class PoseAngle:

    def __init__(self, image, frame_shape, landmarks: list) -> None:
        self.landmarks = landmarks
        self.image = image
        self.frame_shape = frame_shape
        
        self.left_shoulder = [self.landmarks[11].x, self.landmarks[11].y]
        self.left_elbow = [self.landmarks[13].x, self.landmarks[13].y]
        self.left_wrist = [self.landmarks[15].x, self.landmarks[15].y]

        self.right_shoulder = [self.landmarks[12].x, self.landmarks[12].y]
        self.right_elbow = [self.landmarks[14].x, self.landmarks[14].y]
        self.right_wrist = [self.landmarks[16].x, self.landmarks[16].y]

        self.left_elbow_angle_val = self.left_elbow_angle_func()
        self.right_elbow_angle_val = self.right_elbow_angle_func()
        
        self.display_angle(self.image, self.left_elbow_angle_val, self.left_elbow, self.frame_shape)
        self.display_angle(self.image, self.right_elbow_angle_val, self.right_elbow, self.frame_shape)

    def calculate_angle(self, vector_a, vector_b, vector_c):
        a = np.array(vector_a)
        b = np.array(vector_b)
        c = np.array(vector_c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def left_elbow_angle_func(self):
        return self.calculate_angle(self.left_shoulder, self.left_elbow, self.left_wrist)

    def right_elbow_angle_func(self):
        return self.calculate_angle(self.right_shoulder, self.right_elbow, self.right_wrist)

    def display_angle(self, image, angle, joint, frame_shape):
        text = str(int(angle))
        height, width, _ = frame_shape
        position =  tuple(np.multiply(joint, [width, height]).astype(int))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        colour = (255, 255, 255)
        thickness = 2
        
        cv2.putText(image, text, position, font, font_scale, colour, thickness, cv2.LINE_AA)


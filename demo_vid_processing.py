import cv2
import mediapipe as mp
import numpy as np
import csv
from angle_calculation import PoseAngle
mp_drawing = mp.solutions.drawing_utils

demo_video = 'assets/Macarena.mp4'
total_score = 0

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Initialize video capture for demo video
demo_cap = cv2.VideoCapture(demo_video)
angle_joints = ['left_elbow_angle', 'right_elbow_angle', 
                'left_shoulder_angle', 'right_shoulder_angle',
                'left_hip_angle', 'right_hip_angle',
                'left_knee_angle', 'right_knee_angle']

with open('demo_angles.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header: frame number and joint angle indices
    header = []
    for idx in range(len(angle_joints)):
        header.append(angle_joints[idx])
    csv_writer.writerow(header)
    
    frame_index = 0
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while demo_cap.isOpened():
            ret, frame = demo_cap.read()

            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            # Get frame dimensions dynamically
            frame_shape = frame.shape
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            if results.pose_landmarks:
                
                landmarks = results.pose_landmarks.landmark
                
                # Initialize the PoseAngle class (ensure class name matches)
                player = PoseAngle(image, frame_shape, landmarks)

                frame_angles = [player.left_elbow_angle, player.right_elbow_angle,
                                player.left_shoulder_angle, player.right_shoulder_angle,
                                player.left_hip_angle, player.right_hip_angle,
                                player.left_knee_angle, player.right_knee_angle]
                
                # for testing
                # frame_angles = [player.left_elbow, player.right_elbow,
                #                 player.left_shoulder, player.right_shoulder,
                #                 player.left_hip, player.right_hip,
                #                 player.left_knee, player.right_knee,
                #                 player.left_ankle, player.right_ankle]
                # Write angles to CSV
                csv_writer.writerow(frame_angles)
                
                frame_score = player.get_angle_num() * 3
                total_score += frame_score
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,240), thickness=5, circle_radius=10), 
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=5) 
                                 )
        
        
        frame_index += 1
        print(total_score)

# Release capture
demo_cap.release()

import cv2
import mediapipe as mp
import numpy as np
from angle_calculation import PoseAngle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def judge(angle, image, joint1, joint2, frame_shape, standard=90):
    height, width, _ = frame_shape
    
    # Convert normalized landmarks to pixel coordinates
    joint1_pixel = tuple(np.multiply(joint1, [width, height]).astype(int))
    joint2_pixel = tuple(np.multiply(joint2, [width, height]).astype(int))
    
    # Calculate color based on angle
    unit = 255 // 50
    colour_offset = abs(angle - standard) * unit
    R = min(255, colour_offset)
    G = max(0, 255 - colour_offset)
    line_colour = (0, G, R)  # has to be a tuple

    # Draw the line between elbow and wrist
    cv2.line(image, joint1_pixel, joint2_pixel, line_colour, 20)

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,240), thickness=5, circle_radius=10), 
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=5) 
                                 )
        
        try:
            judge(player.left_elbow_angle_val, image,
                  player.left_elbow, player.left_wrist,
                  frame_shape, standard=90)
            judge(player.right_elbow_angle_val, image,
                  player.right_elbow, player.right_wrist,
                  frame_shape, standard=90)
        
        except Exception as e:
            print(e)



        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
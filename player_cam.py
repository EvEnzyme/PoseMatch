import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(vector_a, vector_b, vector_c):
    a = np.array(vector_a)
    b = np.array(vector_b)
    c = np.array(vector_c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def display_angle(image, angle, joint, frame_shape):
    text = str(int(angle))
    height, width, _ = frame_shape
    position =  tuple(np.multiply(joint, [width, height]).astype(int))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    colour = (255, 255, 255)
    thickness = 2
    
    cv2.putText(image, text, position, font, font_scale, colour, thickness, cv2.LINE_AA)

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
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            display_angle(image, angle, elbow, frame_shape)

                       
        except Exception as e:
            print(e)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,240), thickness=5, circle_radius=10), 
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=5) 
                                 )
        
        try:
            judge(angle, image, elbow, wrist,frame_shape, standard=90)
        
        except Exception as e:
            print(e)



        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from angle_calculation import PoseAngle
from scoring import ScoringSystem

demo_video = 'assets/Macarena.mp4'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open the csv file for reading inside the while loop later
csv_file_path = 'demo_angles.csv'
csvfile = open(csv_file_path, newline='')
csv_reader = list(csv.reader(csvfile)) # convert to list for direct indexing lateron
# next(csv_reader) # skip the header

line_index = 1 # start index for reading the csv, =1 to skip the header

# Capture the demo video
demo_cap = cv2.VideoCapture(demo_video)

# Capture the player camera feed
player_cap = cv2.VideoCapture(0)

# declare a list to hold demo angles read from the csv file
demo_angles = []

player_total_score = 0
demo_total_score = 12288 # got from running demo_vid_processing
# TODO: automation in grabbing the demo_total_score

def countdown(player_cap):
    """
    this function is for displaying a 3 second count down at the start of the camera feed
    """
    start_time = time.time()
    countdown_secs = 3  # countdown from 3 seconds

    while True:
        ret_player, player_frame = player_cap.read()
        if not ret_player:
            break

        player_frame = cv2.flip(player_frame, 1)
        player_frame_shape = player_frame.shape

        # Display countdown at the center of the player frame
        elapsed_time = time.time() - start_time
        remaining_time = countdown_secs - int(elapsed_time)

        if remaining_time > 0:
            text = str(remaining_time)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 7
            thickness = 20
            color = (0, 255, 255)  # yellow color for the countdown

            # Calculate position to center the text
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x = (player_frame.shape[1] - text_width) // 2
            y = (player_frame.shape[0] + text_height) // 2

            cv2.putText(player_frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.imshow('Player Feed with Countdown', player_frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

countdown(player_cap)


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:

    # Store the last frame of the demo video
    last_demo_frame = None

    while player_cap.isOpened():
        ret_player, player_frame = player_cap.read()

        if not ret_player:
            break

        # Handle demo video
        ret_demo, demo_frame = demo_cap.read()

        if ret_demo:
            last_demo_frame = demo_frame  # Keep track of the last available frame
        else:
            demo_frame = last_demo_frame  # If the demo video ends, keep showing the last frame



        # Resize demo video to fit in the top-left corner
        demo_frame_resized = cv2.resize(demo_frame, (810, 540))
        
        player_frame = cv2.flip(player_frame, 1)
        # Get player_frame dimensions dynamically
        player_frame_shape = player_frame.shape
        
        # Recolor player_image to RGB
        player_image = cv2.cvtColor(player_frame, cv2.COLOR_BGR2RGB)
        player_image.flags.writeable = False
      
        # Make detection
        results = pose.process(player_image)
    
        # Recolor back to BGR
        player_image.flags.writeable = True
        player_image = cv2.cvtColor(player_image, cv2.COLOR_RGB2BGR)

        # Overlay the demo video on the top-left corner of the player camera feed
        player_image[0:demo_frame_resized.shape[0], 0:demo_frame_resized.shape[1]] = demo_frame_resized

        
        # Extract landmarks
        try:
            
            landmarks = results.pose_landmarks.landmark
            
            # Initialize the PoseAngle class (ensure class name matches)
            player = PoseAngle(player_image, player_frame_shape, landmarks)
            # Render detections
            mp_drawing.draw_landmarks(player_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,240), thickness=5, circle_radius=10), 
                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=5))

            scoring = ScoringSystem(csv_reader, line_index)
            frame_score = scoring.get_frame_score(player, player_image, player_frame_shape)
            print(frame_score)
            player_total_score += frame_score


        except Exception as e:
            print(e)

        # # ensure that the line index for the csv file is not out of range
        # if line_index == len(csv_reader) - 1:
        #         line_index = 1  # loop if reach end of file
        # else:
        line_index += 1
        
        if not ret_demo:
            scoring.display_overall_performance(demo_total_score, player_total_score, player_image)

        cv2.imshow('Player Feed with Demo', player_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    csvfile.close()
    demo_cap.release()
    player_cap.release()
    cv2.destroyAllWindows()
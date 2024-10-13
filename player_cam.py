import cv2
import mediapipe as mp
import numpy as np
import csv
from angle_calculation import PoseAngle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def judge(angle, player_image, joint1, joint2, player_frame_shape, standard=90):
    height, width, _ = player_frame_shape
    
    # Convert normalized landmarks to pixel coordinates
    joint1_pixel = tuple(np.multiply(joint1, [width, height]).astype(int))
    joint2_pixel = tuple(np.multiply(joint2, [width, height]).astype(int))
    
    # Calculate color based on angle
    unit = 255 // 50
    colour_offset = abs(angle - standard) * unit
    R = min(255, colour_offset)
    G = max(0, 255 - colour_offset)
    line_colour = (0, G, R)  # has to be a tuple

    if np.array_equal(joint1, np.zeros(2)) or np.array_equal(joint2, np.zeros(2)):
         return
    
    else:
        # Draw the line between elbow and wrist
        cv2.line(player_image, joint1_pixel, joint2_pixel, line_colour, 20)

# Open the csv file for reading inside the while loop later
csv_file_path = 'demo_angles.csv'
csvfile = open(csv_file_path, newline='')
csv_reader = list(csv.reader(csvfile)) # convert to list for direct indexing lateron
# next(csv_reader) # skip the header

line_index = 1 # start index for reading the csv, =1 to skip the header

# Capture the demo video
demo_cap = cv2.VideoCapture('assets/demo_vid.mov')

# Capture the player camera feed
player_cap = cv2.VideoCapture(0)

# declare a list to hold demo angles read from the csv file
demo_angles = []

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while player_cap.isOpened():
        ret_player, player_frame = player_cap.read()

        if not ret_player:
            break

        # Handle demo video
        ret_demo, demo_frame = demo_cap.read()
        if not ret_demo:
            # If the demo video ends, reset the video capture (loop)
            demo_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_demo, demo_frame = demo_cap.read()

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
                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=5) 
                                    )

            # read a line from the csv file
            csv_row = csv_reader[line_index]

            demo_left_elbow_angle = float(csv_row[0])  # Left elbow angle
            demo_right_elbow_angle = float(csv_row[1])
            demo_left_shoulder_angle = float(csv_row[2])
            demo_right_shoulder_angle = float(csv_row[3])

            # if gets judged every few beats, the standard that's passed into the judge function will need to change
            judge(player.left_elbow_angle, player_image,
                    player.left_elbow, player.left_wrist,
                    player_frame_shape, standard=demo_left_elbow_angle)
            judge(player.right_elbow_angle, player_image,
                    player.right_elbow, player.right_wrist,
                    player_frame_shape, standard=demo_right_elbow_angle)
            judge(player.left_shoulder_angle, player_image,
                    player.left_shoulder, player.left_elbow,
                    player_frame_shape, standard=demo_left_shoulder_angle)
            judge(player.right_shoulder_angle, player_image,
                    player.right_shoulder, player.right_elbow,
                    player_frame_shape, standard=demo_right_shoulder_angle)


        except Exception as e:
            print(e)

        # ensure that the line index for the csv file is not out of range
        if line_index == len(csv_reader) - 1:
                line_index = 1  # loop if reach end of file
        else:
            line_index += 1

        cv2.imshow('Player Feed with Demo', player_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    csvfile.close()
    demo_cap.release()
    player_cap.release()
    cv2.destroyAllWindows()
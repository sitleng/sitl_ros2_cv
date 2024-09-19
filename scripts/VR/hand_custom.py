#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64

import cv2
import math
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


import numpy as np

def compute_angle(base_tip, index_finger_tip, middle_finger_tip):
    x_index=index_finger_tip.x
    y_index=index_finger_tip.y
    p1=[x_index,y_index]

    x_middle=middle_finger_tip.x 
    y_middle=middle_finger_tip.y
    p2=[x_middle,y_middle]

    base_x=base_tip.x
    base_y=base_tip.y
    base=[base_x,base_y]

    vector1 = np.array(p1) - np.array(base)
    vector2 = np.array(p2) - np.array(base)

    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    cosine_angle = dot_product / (norm_vector1 * norm_vector2)
    angle_in_radians = np.arccos(cosine_angle)
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees

def compute_2_norm_distance(index_finger_tip,middle_finger_tip):


    distance = ((middle_finger_tip.x - index_finger_tip.x) ** 2 +
                (middle_finger_tip.y - index_finger_tip.y) ** 2)** 0.5 #+
                #(middle_finger_tip.z - index_finger_tip.z) ** 2) ** 0.5
    
    return distance


def rescale_value(original_value,max_old,min_old):
    min_new = -5
    max_new = -55

    # Rescale the value
    rescaled_value = rescaled_value = ((original_value - min_old) * (max_new - min_new)) / (max_old - min_old) + min_new
    return rescaled_value

'''
def saturation(value,min_v,max_v)
   if value>max_v:
      value=max_v
    
    elif:  
'''
rospy.init_node("pub_hand_angle")

hand_pub = rospy.Publisher("hand_angle",Float64,queue_size=10)

try:

    # For static images:
    IMAGE_FILES = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                cv2.imwrite(
                    '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
                # Draw hand world landmarks.
                if not results.multi_hand_world_landmarks:
                    continue
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(
                        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
            
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    

                    middle_landmark=mp_hands.HandLandmark.MIDDLE_FINGER_TIP 
                    index_landmark=mp_hands.HandLandmark.INDEX_FINGER_TIP
                    
                    middle=hand_landmarks.landmark[middle_landmark]
                    index=hand_landmarks.landmark[index_landmark]
                    wrist=hand_landmarks.landmark[0]

                    angle=compute_angle(wrist,index,middle)
                    
                    angle_norm=rescale_value(angle,16,4)

                    print(angle_norm)

                    msg = Float64()
                    msg.data = angle_norm

                    hand_pub.publish(msg)


                    # Print x and y coordinates of thumbs and indices.
                    for landmark in [mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]:
                        x = hand_landmarks.landmark[landmark].x * image.shape[1]
                        y = hand_landmarks.landmark[landmark].y * image.shape[0]
                        

                        cv2.putText(
                            image_bgr, f"{landmark.name}: ({x:.2f}, {y:.2f})",
                            (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image_bgr)#cv2.flip(image_bgr, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

            
except Exception as e:
    rospy.loginfo(e)


finally:
    cap.release()
    cv2.destroyAllWindows()



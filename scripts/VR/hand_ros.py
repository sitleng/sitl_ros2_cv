#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float64

import cv2
import math
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class VR_HAND():

    def __init__(self):
        
        self.br = CvBridge()
        self.angle_norm = None
        self.angle_norm_yaw = None
        self.image = None 
        self.pub_angle = rospy.Publisher('hand_angle', Float64, queue_size=10)
        self.pub_angle_yaw = rospy.Publisher('hand_angle_yaw', Float64, queue_size=10)

        # Initialize mp_hands.Hands once
        
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )


    def signed_angle(index_finger_tip, middle_finger_tip, base_tip):
        x_index = index_finger_tip.x
        y_index = index_finger_tip.y
        p1 = [x_index, y_index]

        x_middle = middle_finger_tip.x 
        y_middle = middle_finger_tip.y
        p2 = [x_middle, y_middle]

        base_x = base_tip.x
        base_y = base_tip.y
        base = [base_x, base_y]

        vector1 = np.array(p1) - np.array(base)
        vector2 = np.array(p2) - np.array(base)

        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)

        cosine_angle = dot_product / (norm_vector1 * norm_vector2)

        cross_product = np.cross(vector1, vector2)
        angle_in_radians = np.arctan2(cross_product, dot_product)
        angle_in_degrees = np.degrees(angle_in_radians)

        return angle_in_degrees

    def compute_angle(self, base_tip, index_finger_tip, middle_finger_tip):
        x_index = index_finger_tip.x
        y_index = index_finger_tip.y
        p1 = [x_index, y_index]

        x_middle = middle_finger_tip.x
        y_middle = middle_finger_tip.y
        p2 = [x_middle, y_middle]

        base_x = base_tip.x
        base_y = base_tip.y
        base = [base_x, base_y]

        vector1 = np.array(p1) - np.array(base)
        vector2 = np.array(p2) - np.array(base)

        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)

        cosine_angle = dot_product / (norm_vector1 * norm_vector2)
        angle_in_radians = np.arccos(cosine_angle)
        angle_in_degrees = np.degrees(angle_in_radians)

        return angle_in_degrees



    def rescale_value(self, original_value, max_old, min_old,min_new,max_new):


        rescaled_value = ((original_value - min_old) * (max_new - min_new)) / (max_old - min_old) + min_new
        if rescaled_value > max_new:
            rescaled_value = max_new
        elif rescaled_value < min_new:
            rescaled_value = min_new
        return rescaled_value

    def process_image(self):
        if self.image is not None:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        self.image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    middle_landmark = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    index_landmark = mp_hands.HandLandmark.INDEX_FINGER_TIP
                    middle = hand_landmarks.landmark[middle_landmark]
                    index = hand_landmarks.landmark[index_landmark]
                    wrist = hand_landmarks.landmark[0]
                    index_mcp=hand_landmarks.landmark[5]
                    pinky_mcp=hand_landmarks.landmark[17]

                    angle_yaw_signed = self.signed_angle(index_mcp, pinky_mcp, wrist)
                    self.angle_norm_yaw = self.rescale_value(angle_yaw_signed, -32, 32,-200,200)

                    angle = self.compute_angle(wrist, index, middle)
                    self.angle_norm = self.rescale_value(angle, 16, 4,-5,55)

                    # Print x and y coordinates of thumbs and indices.
                    for landmark in [mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]:
                        x = hand_landmarks.landmark[landmark].x * self.image.shape[1]
                        y = hand_landmarks.landmark[landmark].y * self.image.shape[0]

                        cv2.putText(
                            self.image, f"{landmark.name}: ({x:.2f}, {y:.2f})",
                            (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

            if self.angle_norm is not None:
                self.pub_angle.publish(self.angle_norm)
            if self.angle_norm_yaw is not None:
                self.pub_angle_yaw.publish(self.angle_norm_yaw)

    def callback(self, imgmsg):
        self.image = self.br.compressed_imgmsg_to_cv2(imgmsg)
        self.process_image()

if __name__ == '__main__':
    rospy.init_node('hand_gesture_node', anonymous=True)
    hand_app = VR_HAND()
    
    try:
        rate = rospy.Rate(30)  # 10 Hz
        rospy.Subscriber("/vr/left/image_color", CompressedImage, hand_app.callback)

        while not rospy.is_shutdown():
            if hand_app.image is not None:
                cv2.imshow('MediaPipe Hands', hand_app.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop on 'Esc' key
                    break
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

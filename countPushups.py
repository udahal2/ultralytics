import cv2
import mediapipe as mp
import numpy as np
import torch
import time

# Use the YOLOv8 model (YOLOv11 can be similar if you've set it up similarly)
class PushUpCounter:
    def __init__(self, range_of_motion_threshold=30, feedback_color=(0, 255, 0), count_display=True):
        self.range_of_motion_threshold = range_of_motion_threshold
        self.feedback_color = feedback_color
        self.count_display = count_display

        self.pushup_count = 0
        self.in_pushup = False
        self.prev_frame_time = 0
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = torch.hub.load("ultralytics/yolov5", "v5.0")  # Replace with YOLOv8 or YOLOv11 if available
        self.cap = cv2.VideoCapture(0)  # Use webcam

    def process_frame(self, frame):
        # Run YOLO detection (detect people and objects)
        results = self.model(frame)
        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use Mediapipe to find pose landmarks
        results_pose = self.pose.process(frame_rgb)
        
        return results, results_pose, frame
    
    def calculate_arm_angle(self, elbow, shoulder, wrist):
        # Calculate the angle between shoulder, elbow, and wrist
        a = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
        b = np.array([wrist.x - shoulder.x, wrist.y - shoulder.y])
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def count_pushups(self, landmarks):
        if landmarks and landmarks.pose_landmarks:
            # Extract relevant pose landmarks (elbows, shoulders, etc.)
            left_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate arm angles
            left_angle = self.calculate_arm_angle(left_elbow, left_shoulder, left_wrist)
            right_angle = self.calculate_arm_angle(right_elbow, right_shoulder, right_wrist)

            # Determine the range of motion and count pushups
            if left_angle < self.range_of_motion_threshold and right_angle < self.range_of_motion_threshold:
                if not self.in_pushup:
                    self.pushup_count += 1
                    self.in_pushup = True
            elif left_angle > 160 and right_angle > 160:
                self.in_pushup = False

    def display_pushup_count(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        count_text = f"Push-ups: {self.pushup_count}"
        cv2.putText(frame, count_text, (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                    font, 1, self.feedback_color, 2, cv2.LINE_AA)

    def start_counter(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process the frame for object and pose detection
            results, results_pose, frame = self.process_frame(frame)

            # Count push-ups based on pose landmarks
            self.count_pushups(results_pose)

            # Display push-up count on the frame
            if self.count_display:
                self.display_pushup_count(frame)

            # Display the resulting frame
            cv2.imshow('Push-Up Counter', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = PushUpCounter()
    counter.start_counter()

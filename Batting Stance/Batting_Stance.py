import cv2
import mediapipe as mp
import numpy as np


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle


# Function to detect bat position
def detect_bat_position(landmarks):
    if len(landmarks) >= 6:  # Ensuring we have enough points for calculations
        angle_left = calculate_angle(landmarks[11], landmarks[13], landmarks[15])  # Angle of left arm
        angle_right = calculate_angle(landmarks[12], landmarks[14], landmarks[16])  # Angle of right arm
        avg_angle = (angle_left + angle_right) / 2  # Average angle of arms
        if avg_angle > 170:  # Bat is kept too high
            return "Low"
        elif avg_angle < 100:  # Bat is kept too low
            return "High"
    return "Normal"


# Function to draw pose landmarks on the frame
def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        cv2.circle(frame, landmark, 5, (0, 255, 0), -1)


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    results = pose.process(frame_rgb)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmarks.append((cx, cy))

        # Detect bat position
        bat_position = detect_bat_position(landmarks)

        # Draw pose landmarks
        draw_landmarks(frame, landmarks)

        # Display bat position
        cv2.putText(frame, f"Bat Position: {bat_position}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Human Pose Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

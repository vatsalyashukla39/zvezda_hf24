import tensorflow as tf
import tensorflow_hub as hub
import cv2
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from joblib import dump
import mediapipe as mp


mp_pose = mp.solutions.pose.Pose()


# Load MoveNet model
movenet = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/1')
movenet = movenet.signatures['serving_default']

def process_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return image


def get_pose_data(image):
    results = mp_pose.process(image)
    if results.pose_landmarks is None:
        print("No pose landmarks detected in the image.")
        return [0]*132  # Return a list of zeros of the appropriate length
    keypoints = results.pose_landmarks.landmark
    keypoints_list = [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in keypoints]

    # Print the coordinates of the joints
    for i, keypoint in enumerate(keypoints_list):
        print(f'Keypoint {i}: x={keypoint[0]}, y={keypoint[1]}, z={keypoint[2]}, visibility={keypoint[3]}')

    return keypoints_list

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def stretch_video(video_path, desired_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ratio = total_frames / desired_frames

    stretched_frames = []
    for i in range(desired_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(i * frame_ratio))
        ret, frame = cap.read()
        if ret:
            stretched_frames.append(frame)

    cap.release()
    return stretched_frames


def extract_pose_data(video_path, desired_frames, label):
    stretched_frames = stretch_video(video_path, desired_frames)

    pose_data = []
    for frame in stretched_frames:
        image = process_frame(frame)
        keypoints = get_pose_data(image)
        # Flatten the keypoints
        keypoints = np.array(keypoints).reshape(-1)
        pose_data.append((keypoints.tolist(), label))

    return pose_data

def calculate_average_keypoints(video_dir, desired_frames, label):
    video_files = [f for f in os.listdir(video_dir) if f.startswith(label) and f.endswith('.mp4')]

    sum_keypoints = None
    for video_file in tqdm(video_files, desc=f"Processing {label} videos"):
        video_path = os.path.join(video_dir, video_file)
        pose_data = extract_pose_data(video_path, desired_frames, label)
        keypoints = np.array([data[0] for data in pose_data])
        if sum_keypoints is None:
            sum_keypoints = keypoints
        else:
            sum_keypoints += keypoints

    average_keypoints = sum_keypoints / len(video_files)
    return average_keypoints

def extract_all_videos(video_dir, desired_frames):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    all_pose_data = []
    for video_file in tqdm(video_files, desc="Processing videos"):
        label = video_file.split('_')[0]  # Extract label from filename
        video_path = os.path.join(video_dir, video_file)
        pose_data = extract_pose_data(video_path, desired_frames, label)
        all_pose_data.extend(pose_data)

    return all_pose_data

# Calculate the average keypoints for the "straight" videos
average_straight_keypoints = calculate_average_keypoints('data', 30, 'straightDrive')
average_cover_keypoints = calculate_average_keypoints('data', 30, 'coverDrive')

# Save the average keypoints to a file
np.save('average_straight_keypoints.npy', average_straight_keypoints)

np.save('average_cover_keypoints.npy', average_cover_keypoints)

# Collect all pose data
all_pose_data = extract_all_videos('data', 30)

# Split data into features and labels
X = [data[0] for data in all_pose_data]  # features
y = [data[1] for data in all_pose_data]  # labels

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)


# Save the model to a file
dump(model, 'model.joblib')

# Evaluate the model
print(f'Model accuracy: {model.score(X_test, y_test)}')
print(f'Cross-validation scores: {scores}')
print(f'Average score: {np.mean(scores)}')
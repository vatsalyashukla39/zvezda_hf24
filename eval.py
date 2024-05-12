import numpy as np
from joblib import load
from scipy import stats


from main import extract_pose_data

# Load the model from the file
model = load('model.joblib')

from joblib import load

# Load the model from the file
model = load('model.joblib')

# Define a function to process the user input video and extract pose data
def process_user_video(video_path):
    # Replace this with the actual number of frames you want to extract from the video
    desired_frames = 52

    # Extract pose data from the video
    pose_data = extract_pose_data(video_path, desired_frames, label=None)

    # Flatten the pose data and convert it to a numpy array
    pose_data = np.array([data[0] for data in pose_data])

    return pose_data

keypoints_mapping = {
    0: 'nose',
    1: 'leftEye',
    2: 'rightEye',
    3: 'leftEar',
    4: 'rightEar',
    5: 'leftShoulder',
    6: 'rightShoulder',
    7: 'leftElbow',
    8: 'rightElbow',
    9: 'leftWrist',
    10: 'rightWrist',
    11: 'leftHip',
    12: 'rightHip',
    13: 'leftKnee',
    14: 'rightKnee',
    15: 'leftAnkle',
    16: 'rightAnkle'
}


# Process the user input video
video_path = 'data/notstraight_1.mp4'
pose_data = process_user_video(video_path)

# Load the average keypoints for the "straight" videos
average_straight_keypoints = np.load('average_straight_keypoints.npy')

# Calculate the difference between the user's keypoints and the average "straight" keypoints
keypoint_differences = np.abs(pose_data - average_straight_keypoints)

# Find the index of the keypoint with the largest difference
most_different_keypoint_index = np.argmax(keypoint_differences)



# Check if pose_data is empty
if pose_data.size == 0:
    print("No pose data extracted from the video.")
else:
    # Reshape pose_data to 2D if it contains only one element
    if pose_data.ndim == 1:
        pose_data = np.reshape(pose_data, (1, -1))


    # Use the model to predict the label for each frame in the pose data
    predicted_labels = model.predict(pose_data)

    # Take the mode of the predicted labels
    overall_predicted_label = stats.mode(predicted_labels)[0]

    # Print the overall predicted label
    print(f'The overall predicted label for the video is: {overall_predicted_label}')

# Print the index of the most different keypoint
print(f'The joint that need the  most improvement is joint {improv}.')
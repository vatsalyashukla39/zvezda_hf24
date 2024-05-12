import numpy as np
import cv2
from joblib import load
from scipy import stats
import os


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

def play_video(video_path):
    # Check if the file exists
    if not os.path.isfile(video_path):
        print(f"File {video_path} does not exist.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


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
    overall_predicted_label = stats.mode(predicted_labels)[0][0]

    # Print the overall predicted label
    print(f'The overall predicted label for the video is: {overall_predicted_label}')

# Calculate the difference between the user's keypoints and the average "straight" keypoints
keypoint_differences = np.abs(pose_data - average_straight_keypoints)

# Find the indices that would sort the keypoint differences
sorted_keypoint_indices = np.argsort(keypoint_differences.flatten())

# Get the indices of the keypoints with the largest, second largest, and third largest differences
most_different_keypoint_index = sorted_keypoint_indices[-1] % 68 // 4
second_most_different_keypoint_index = sorted_keypoint_indices[-2] % 68 // 4
third_most_different_keypoint_index = sorted_keypoint_indices[-3] % 68 // 4


# Get the names of the joints that correspond to these keypoints
most_different_joint = keypoints_mapping[most_different_keypoint_index]
second_most_different_joint = keypoints_mapping[second_most_different_keypoint_index]
third_most_different_joint = keypoints_mapping[third_most_different_keypoint_index]



# # Map the index back to the corresponding keypoint
# most_different_keypoint_index = most_different_keypoint_index % 68 // 4

# # Get the name of the joint that corresponds to the most different keypoint
# most_different_joint = keypoints_mapping[most_different_keypoint_index]


# Print the index of the most different keypoint
# print(f'The joint that need the  most improvement is joint {improv}.')

# Play the video if the overall predicted label is 'notstraight'
if overall_predicted_label == 'notstraight':
    print("The shot you played is not close to a standard straight drive. Please work on the following points:")
    #print(f"Most different keypoint: {keypoints_mapping[most_different_keypoint_index]}")
    print(" Hold the bat in a way that the top hand grips the handle of the bat comfortably, while the bottom hand provides it support and control")
    print("Please watch the video for further analysis.")
    # play_video('data/notstraight_1.mp4')
    # Print the name of the most different joint
    # Print the names of the joints that need the most improvement
    print(f'The joint that needs the most improvement is: {most_different_joint}')
    print(f'The joint that needs the second most improvement is: {second_most_different_joint}')
    print(f'The joint that needs the third most improvement is: {third_most_different_joint}')

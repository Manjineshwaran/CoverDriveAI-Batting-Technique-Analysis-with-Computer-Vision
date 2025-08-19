import cv2
import mediapipe as mp
import numpy as np

# ------------------- Mediapipe setup -------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ------------------- Thresholds -------------------
MIN_FEET = 0.1      # Adjust according to frame size / normalized coordinates
MAX_FEET = 0.35
MIN_BEND = 0.05
MAX_BEND = 0.2
MOVEMENT_THRESHOLD = 0.03
BALANCE_THRESHOLD = 0.05

# ------------------- Utility Functions -------------------
def compute_movement(pose1, pose2):
    if pose1 is None:
        return 0
    movement = 0
    for k in pose1:
        movement += abs(pose1[k].x - pose2[k].x) + abs(pose1[k].y - pose2[k].y)
    return movement

def is_stance_bowler_view(pose_landmarks, prev_pose=None):
    keypoints = {lm.name: lm for lm in mp_pose.PoseLandmark}
    # Get coordinates
    left_ankle = pose_landmarks[keypoints['LEFT_ANKLE']].normalized
    right_ankle = pose_landmarks[keypoints['RIGHT_ANKLE']].normalized
    left_knee = pose_landmarks[keypoints['LEFT_KNEE']].normalized
    right_knee = pose_landmarks[keypoints['RIGHT_KNEE']].normalized
    left_hip = pose_landmarks[keypoints['LEFT_HIP']].normalized
    right_hip = pose_landmarks[keypoints['RIGHT_HIP']].normalized
    left_wrist = pose_landmarks[keypoints['LEFT_WRIST']].normalized
    right_wrist = pose_landmarks[keypoints['RIGHT_WRIST']].normalized
    nose = pose_landmarks[keypoints['NOSE']].normalized

    # 1. Feet distance
    foot_dist = abs(left_ankle.x - right_ankle.x)
    if not (MIN_FEET < foot_dist < MAX_FEET):
        return False

    # 2. Knee bend (slight squat)
    hip_y = (left_hip.y + right_hip.y) / 2
    ankle_y = (left_ankle.y + right_ankle.y) / 2
    knee_bend = hip_y - ankle_y
    if not (MIN_BEND < knee_bend < MAX_BEND):
        return False

    # 3. Bat behind check (wrist vs hip)
    bat_x = (left_wrist.x + right_wrist.x) / 2
    hip_x = (left_hip.x + right_hip.x) / 2
    if bat_x > hip_x:  # Bat swung forward
        return False

    # 4. Minimal movement
    if prev_pose is not None:
        movement = compute_movement(pose_landmarks, prev_pose)
        if movement > MOVEMENT_THRESHOLD:
            return False

    # 5. Head orientation
    if nose.x < min(left_ankle.x, right_ankle.x) or nose.x > max(left_ankle.x, right_ankle.x):
        return False

    # 6. Weight balance (CoG midline)
    cog_x = (left_hip.x + right_hip.x) / 2
    mid_feet_x = (left_ankle.x + right_ankle.x) / 2
    if abs(cog_x - mid_feet_x) > BALANCE_THRESHOLD:
        return False

    return True

# ------------------- Video Processing -------------------
input_video = "D:/AIDS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/data/input_video.mp4"
output_video = "output_stance_video.mp4"

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

prev_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    stance_detected = False
    if results.pose_landmarks:
        # Convert landmarks to dict
        landmarks_dict = {lm.name: lm for lm in mp_pose.PoseLandmark}
        pose_landmarks = results.pose_landmarks.landmark
        stance_detected = is_stance_bowler_view(pose_landmarks, prev_landmarks)
        prev_landmarks = pose_landmarks

        # Draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display label
    label = "STANCE" if stance_detected else "NOT STANCE"
    cv2.putText(frame, label, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0) if stance_detected else (0,0,255), 3)

    out.write(frame)
    cv2.imshow("Stance Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# cover_drive_analysis_realtime.py

import cv2
import mediapipe as mp
import numpy as np
import os
import math

# ---------------------------
# Step 0: Setup & Config
# ---------------------------

VIDEO_PATH = "D:/AI&DS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/data/extracted_frames_video.mp4"
OUTPUT_FOLDER = "D:/AI&DS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# Step 1: Load Video
# ---------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise IOError(f"Cannot open video file at {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height} at {fps} FPS, {frame_count} frames total")

# ---------------------------
# Step 2: Setup VideoWriter
# ---------------------------

output_path = os.path.join(OUTPUT_FOLDER, "annotated_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ---------------------------
# Step 3: Pose Estimation Setup
# ---------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# Helper Functions
# ---------------------------

def calculate_angle(a, b, c):
    """Calculate angle between three points a-b-c in degrees"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return int(np.degrees(angle))

def overlay_metrics(frame, metrics):
    """Draw metrics and feedback on frame"""
    annotated_frame = frame.copy()
    y0 = 50
    dy = 30

    # Display numeric metrics
    cv2.putText(annotated_frame, f"Elbow: {metrics['elbow_angle']}°", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(annotated_frame, f"Spine Lean: {metrics['spine_lean']}°", (10, y0+dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(annotated_frame, f"Head-Knee Dist: {metrics['head_knee_dist']}", (10, y0+2*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Feedback
    feedback_y = y0 + 3*dy
    if metrics['elbow_angle'] > 110:
        cv2.putText(annotated_frame, "✅ Good elbow elevation", (10, feedback_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(annotated_frame, "❌ Elbow too low", (10, feedback_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    feedback_y += dy
    if metrics['head_knee_dist'] < 20:  # pixel threshold
        cv2.putText(annotated_frame, "✅ Head over knee", (10, feedback_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(annotated_frame, "❌ Head not over knee", (10, feedback_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return annotated_frame

# ---------------------------
# Step 4: Frame-by-Frame Processing
# ---------------------------

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose estimation
    results = pose.process(image_rgb)

    # Annotate frame
    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    metrics = {"elbow_angle": 0, "spine_lean": 0, "head_knee_dist": 0}

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract keypoints
        keypoints = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            x, y = int(lm.x * width), int(lm.y * height)
            keypoints[idx] = (x, y)
            if idx < 5:
                cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)

        # ---------------------------
        # Simple metrics calculation
        # ---------------------------

        # Left elbow angle: Shoulder(11)-Elbow(13)-Wrist(15)
        if 11 in keypoints and 13 in keypoints and 15 in keypoints:
            metrics['elbow_angle'] = calculate_angle(keypoints[11], keypoints[13], keypoints[15])

        # Spine lean: angle between shoulders line and vertical (approx)
        if 11 in keypoints and 12 in keypoints:
            dx = keypoints[12][0] - keypoints[11][0]
            dy = keypoints[12][1] - keypoints[11][1]
            metrics['spine_lean'] = int(math.degrees(math.atan2(dy, dx)))

        # Head over knee distance: Nose(0) vs left knee(25)
        if 0 in keypoints and 25 in keypoints:
            metrics['head_knee_dist'] = abs(keypoints[0][0] - keypoints[25][0])

        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx} metrics: {metrics}")

    # Overlay metrics & feedback
    annotated_frame = overlay_metrics(annotated_frame, metrics)

    # Write and show
    out.write(annotated_frame)
    cv2.imshow("Pose + Metrics", annotated_frame)

    # Pause or quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(0)

# ---------------------------
# Step 5: Cleanup
# ---------------------------

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
print(f"Annotated video saved at {output_path}")

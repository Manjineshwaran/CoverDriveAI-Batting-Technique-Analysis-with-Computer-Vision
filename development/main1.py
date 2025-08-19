# cover_drive_analysis_realtime.py

import cv2
import mediapipe as mp
import numpy as np
import os

# ---------------------------
# Step 0: Setup & Config
# ---------------------------

VIDEO_PATH = "D:/AIDS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/data/input_video.mp4"
OUTPUT_FOLDER = "D:/AIDS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/output"

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

    # Annotate frame for debugging
    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw first 5 keypoints as red circles
        keypoints = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            x, y = int(lm.x * width), int(lm.y * height)
            keypoints[idx] = (x, y)
            if idx < 5:
                cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
        
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx} keypoints (first 5): {list(keypoints.items())[:5]}")

    # Show annotated frame
    cv2.imshow("Pose Debug View", annotated_frame)

    # Write annotated frame to output video
    out.write(annotated_frame)

    # Press 'q' to quit or 'p' to pause frame-by-frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(0)  # pause until any key is pressed

# ---------------------------
# Step 5: Cleanup
# ---------------------------

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
print(f"Annotated video saved at {output_path}")

import cv2
import numpy as np
import os

# ----------------- Config -----------------
input_video = "D:/AIDS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/data/input_video.mp4"
output_video = "D:/AIDS/cv_job_assignment/extracted_frames_video.mp4"

# ----------------- Open video -----------------
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"Error: Cannot open video file: {input_video}")
    exit()

# ----------------- Video properties -----------------
fps = cap.get(cv2.CAP_PROP_FPS)
fps = max(1, int(fps))  # Ensure at least 1
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ----------------- Create VideoWriter -----------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: VideoWriter failed to open.")
    cap.release()
    exit()

# ----------------- Select 3 evenly spaced frames -----------------
frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)

# ----------------- Extract and write frames -----------------
for frame_idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        # Write each frame for ~1 second duration
        for _ in range(fps):
            out.write(frame)
    else:
        print(f"Warning: Failed to read frame {frame_idx}")

# ----------------- Cleanup -----------------
cap.release()
out.release()

if os.path.exists(output_video):
    print(f"Saved video successfully: {output_video}")
else:
    print("Error: Video not saved.")

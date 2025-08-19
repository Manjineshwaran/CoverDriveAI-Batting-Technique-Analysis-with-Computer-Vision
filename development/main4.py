# cover_drive_analysis_realtime.py

import cv2
import mediapipe as mp
import numpy as np
import os
import math
import json
import urllib.request
from typing import Dict, List

# ---------------------------
# Step 0: Setup & Config
# ---------------------------

# YouTube video download (added)
VIDEO_URL = "https://youtube.com/shorts/vSX3IRxGnNY"
OUTPUT_FOLDER = "./output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Download video (new)
def download_youtube_video(url: str, save_path: str) -> str:
    """Mock downloader - in practice use yt-dlp/pytube"""
    # In a real implementation, use:
    # !yt-dlp -f 'bestvideo[ext=mp4]' -o temp_video.mp4 {url}
    # For now we'll assume you've manually downloaded the video
    return "temp_video.mp4"

# Download video if not exists (comment out if using local)
# LOCAL_VIDEO_PATH = download_youtube_video(VIDEO_URL, "input_video.mp4")
LOCAL_VIDEO_PATH = "D:/AIDS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/data/input_video.mp4"
# ---------------------------
# Step 1: Load Video
# ---------------------------

cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file at {LOCAL_VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
print("fps :", fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("width :", width)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("height :", height)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame_count :", frame_count)

print(f"Video info: {width}x{height} at {fps:.1f} FPS, {frame_count} frames total")

# Normalize FPS if needed (added)
TARGET_FPS = 30
if fps > TARGET_FPS:
    print(f"Downsampling FPS from {fps} to {TARGET_FPS}")
    fps = TARGET_FPS

# ---------------------------
# Step 2: Setup VideoWriter
# ---------------------------

output_path = os.path.join(OUTPUT_FOLDER, "annotated_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ---------------------------
# Step 3: Pose Estimation Setup
# ---------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# Helper Functions (Enhanced)
# ---------------------------

def calculate_angle(a, b, c) -> float:
    """Calculate angle between three points a-b-c in degrees with occlusion check"""
    if None in (a, b, c):
        return None
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)

def get_foot_direction(ankle: tuple, toe: tuple, frame_width: int) -> float:
    """Calculate foot angle relative to crease (simplified as x-axis)"""
    if None in (ankle, toe):
        return None
    dx = toe[0] - ankle[0]
    dy = toe[1] - ankle[1]
    return np.degrees(np.arctan2(dy, dx))

def overlay_metrics(frame, metrics: Dict) -> np.ndarray:
    """Enhanced overlay with all metrics and foot direction"""
    annotated_frame = frame.copy()
    y0, dy = 50, 30
    
    # Metric display order
    metric_texts = [
        f"Elbow: {metrics.get('elbow_angle', 'N/A')}°",
        f"Spine: {metrics.get('spine_lean', 'N/A')}°",
        f"Head-Knee: {metrics.get('head_knee_dist', 'N/A')}px",
        f"Foot Angle: {metrics.get('foot_angle', 'N/A')}°"
    ]
    
    for i, text in enumerate(metric_texts):
        cv2.putText(annotated_frame, text, (10, y0+i*dy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Feedback system (enhanced)
    feedback = []
    if metrics.get('elbow_angle', 0) > 110:
        feedback.append("✅ Good elbow")
    else:
        feedback.append("❌ Low elbow")
        
    if abs(metrics.get('foot_angle', 0)) < 30:  # Foot parallel to crease
        feedback.append("✅ Foot aligned")
    else:
        feedback.append("❌ Foot misaligned")
    
    for i, fb in enumerate(feedback):
        cv2.putText(annotated_frame, fb, (width-200, y0+i*dy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if "✅" in fb else (0, 0, 255), 2)
    
    return annotated_frame

# ---------------------------
# Step 4: Frame Processing (Enhanced)
# ---------------------------

frame_idx = 0
all_metrics = []
keypoint_history = []  # For occlusion handling

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Pose estimation
    results = pose.process(image_rgb)
    # print("Pose estimation results: ", results)
    # print("Pose estimation results: ", results.pose_landmarks)
    # print("Pose estimation results: ", results.pose_landmarks.landmark)
    # print(f"Pose results available: {bool(results.pose_landmarks)}")


    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, f"Frame: {frame_idx}/{frame_count}", (width-200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("annotated_frame", annotated_frame)
    cv2.waitKey(500)
    
    metrics = {
        "elbow_angle": None,
        "spine_lean": None,
        "head_knee_dist": None,
        "foot_angle": None
    }
    
    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        # Extract keypoints with confidence check
        keypoints = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > 0.5:  # Confidence threshold
                keypoints[idx] = (int(lm.x * width), int(lm.y * height))
        print(f"Keypoints: {keypoints}")

        # Store for occlusion handling
        keypoint_history.append(keypoints)
        if len(keypoint_history) > 5:
            keypoint_history.pop(0)
        
        # ---------------------------
        # Metric Calculations (Enhanced)
        # ---------------------------
        
        # Elbow angle (using left arm)
        if all(k in keypoints for k in [11, 13, 15]):  # Shoulder, elbow, wrist
            metrics['elbow_angle'] = calculate_angle(
                keypoints[11], keypoints[13], keypoints[15])
        
        # Spine lean
        if all(k in keypoints for k in [11, 12, 23, 24]):  # Shoulders and hips
            shoulder_mid = ((keypoints[11][0] + keypoints[12][0]) / 2,
                           (keypoints[11][1] + keypoints[12][1]) / 2)
            hip_mid = ((keypoints[23][0] + keypoints[24][0]) / 2,
                      (keypoints[23][1] + keypoints[24][1]) / 2)
            metrics['spine_lean'] = np.degrees(
                np.arctan2(hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]))
        
        # Head-knee alignment
        if all(k in keypoints for k in [0, 25]):  # Nose and left knee
            metrics['head_knee_dist'] = abs(keypoints[0][0] - keypoints[25][0])
        
        # Foot direction (NEW) - using ankle and foot index
        if all(k in keypoints for k in [13, 15, 19, 21]):  # Ankle and toe
            # Use left foot (assuming front foot)
            metrics['foot_angle'] = get_foot_direction(
                keypoints[13], keypoints[15], width)
    
    # ---------------------------
    # Step 5: Overlay & Debug
    # ---------------------------
    annotated_frame = overlay_metrics(annotated_frame, metrics)
    all_metrics.append(metrics)
    
    # Debug display (always show)
    cv2.imshow("Debug View", annotated_frame)
    out.write(annotated_frame)
    
    # Frame control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        while cv2.waitKey(1) != ord('p'):
            pass

# ---------------------------
# Final Evaluation (Enhanced)
# ---------------------------

def compute_final_scores(metrics_list: List[Dict]) -> Dict:
    """Enhanced scoring with foot direction"""
    # Filter valid metrics
    valid_metrics = {
        'elbow': [m['elbow_angle'] for m in metrics_list if m['elbow_angle'] is not None],
        'spine': [m['spine_lean'] for m in metrics_list if m['spine_lean'] is not None],
        'head_knee': [m['head_knee_dist'] for m in metrics_list if m['head_knee_dist'] is not None],
        'foot': [m['foot_angle'] for m in metrics_list if m['foot_angle'] is not None]
    }
    print(f"\n Valid Metrics: {valid_metrics}")
    def normalize_score(val, ideal, tolerance):
        return max(1, min(10, int(10 * (1 - abs(val - ideal) / tolerance))))
    
    avg_elbow = np.mean(valid_metrics['elbow']) if valid_metrics['elbow'] else 0
    avg_foot = np.mean(valid_metrics['foot']) if valid_metrics['foot'] else 0
    
    return {
        "Footwork": {
            "score": normalize_score(avg_foot, 0, 30),
            "feedback": "Front foot could open more" if avg_foot > 15 else "Good foot placement"
        },
        "Head Position": {
            "score": normalize_score(np.mean(valid_metrics['head_knee']), 10, 20),
            "feedback": "Maintain head over front knee"
        },
        "Swing Control": {
            "score": normalize_score(avg_elbow, 120, 30),
            "feedback": "Compact backlift improves control"
        },
        "Balance": {
            "score": normalize_score(np.mean(np.abs(valid_metrics['spine'])), 5, 10),
            "feedback": "Excellent weight transfer"
        },
        "Follow-through": {
            "score": normalize_score(avg_elbow, 150, 40),
            "feedback": "Complete your swing fully"
        }
    }

print("\n all_metrics: ", all_metrics[0:2])
evaluation = compute_final_scores(all_metrics)
eval_path = os.path.join(OUTPUT_FOLDER, "evaluation.json")
with open(eval_path, 'w') as f:
    json.dump(evaluation, f, indent=2)

print(f"\nProcessing complete!")
print(f"Annotated video: {output_path}")
print(f"Evaluation saved: {eval_path}")

# ---------------------------
# Cleanup
# ---------------------------

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
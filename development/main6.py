# cover_drive_analysis_realtime.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------- Utility Functions ----------
def angle(a, b, c):
    """Return angle (in degrees) at b given 3 points (x,y)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# ---------- Main Analysis ----------
def analyze_video(path, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))

    out = cv2.VideoWriter(f"{out_dir}/annotated_video.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    metrics_log = []   # store [elbow, spine, head_dist, foot_angle]
    checks_log = []    # store framewise OK/Not for evaluation
    all_frame_values = []  # Variable to store all each frame values

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                h_, w_ = frame.shape[:2]

                # Get key joints
                def pt(idx): return (lm[idx].x * w_, lm[idx].y * h_)
                L_sh, L_el, L_wr = pt(11), pt(13), pt(15)
                R_sh, R_el, R_wr = pt(12), pt(14), pt(16)
                L_hip, R_hip = pt(23), pt(24)
                L_knee, R_knee = pt(25), pt(26)
                L_ankle, R_ankle = pt(27), pt(28)
                L_heel, R_heel = pt(29), pt(30)
                L_toe, R_toe = pt(31), pt(32)
                nose = pt(0)

                # --- Metrics ---
                elbow_angle = angle(L_sh, L_el, L_wr)
                hip_mid = ((L_hip[0] + R_hip[0]) / 2, (L_hip[1] + R_hip[1]) / 2)
                shoulder_mid = ((L_sh[0] + R_sh[0]) / 2, (L_sh[1] + R_sh[1]) / 2)
                vertical_ref = (hip_mid[0], hip_mid[1] - 100)
                spine_angle = angle(vertical_ref, hip_mid, shoulder_mid)
                head_knee_dist = abs(nose[0] - L_knee[0])
                foot_angle = angle((L_heel[0] - 100, L_heel[1]), L_heel, L_toe)

                metrics_log.append([elbow_angle, spine_angle, head_knee_dist, foot_angle])

                # --- Checks ---
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                checks = {}

                # Footwork check (whole video)
                checks["foot_ok"] = (60 < foot_angle < 120)

                # Other checks only for first 3 seconds
                if frame_idx < fps * 3:
                    checks["head_ok"] = (head_knee_dist < 50)
                    checks["balance_ok"] = (abs(spine_angle) < 20)
                    checks["elbow_ok"] = (100 < elbow_angle < 140)

                checks_log.append(checks)

                # Store all frame values - convert boolean values to strings for JSON serialization
                frame_data = {
                    "frame_idx": frame_idx,
                    "metrics": [elbow_angle, spine_angle, head_knee_dist, foot_angle],
                    "checks": {k: str(v) for k, v in checks.items()},  # Convert boolean to string
                    "landmarks": {i: (lm[i].x, lm[i].y, lm[i].z) for i in range(len(lm))}
                }
                all_frame_values.append(frame_data)

                # --- Draw pose + metrics ---
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                cv2.putText(frame, f"Spine Lean: {int(spine_angle)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                cv2.putText(frame, f"Head-Knee Xdist: {int(head_knee_dist)}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                cv2.putText(frame, f"Foot Dir: {int(foot_angle)}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

                if frame_idx < fps * 3:
                    cv2.putText(frame, f"Head Pos: {'OK' if checks['head_ok'] else 'Wrong'}", (10,170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0) if checks['head_ok'] else (0,0,255),2)
                    cv2.putText(frame, f"Balance: {'OK' if checks['balance_ok'] else 'Off'}", (10,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0) if checks['balance_ok'] else (0,0,255),2)
                    cv2.putText(frame, f"Elbow: {'OK' if checks['elbow_ok'] else 'Wrong'}", (10,230),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0) if checks['elbow_ok'] else (0,0,255),2)

                cv2.imshow("frame", frame)
                out.write(frame)

                # Controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    while cv2.waitKey(1) != ord('p'):
                        pass

    cap.release()
    out.release()

    # --- Final cumulative scoring ---
    if metrics_log:
        total_frames = len(metrics_log)

        scores = {}
        feedback = {}

        # Footwork (whole video)
        foot_ok = sum(1 for c in checks_log if "foot_ok" in c and c["foot_ok"])
        scores["Footwork"] = int((foot_ok / total_frames) * 10)
        feedback["Footwork"] = f"Footwork OK in {foot_ok}/{total_frames} frames."

        # Head Position (first 3s only)
        head_checks = [c["head_ok"] for c in checks_log if "head_ok" in c]
        if head_checks:
            head_ok = sum(head_checks)
            scores["Head Position"] = int((head_ok / len(head_checks)) * 10)
            feedback["Head Position"] = f"Head aligned in {head_ok}/{len(head_checks)} frames (first 3s)."

        # Swing Control (elbow) (first 3s only)
        elbow_checks = [c["elbow_ok"] for c in checks_log if "elbow_ok" in c]
        if elbow_checks:
            elbow_ok = sum(elbow_checks)
            scores["Swing Control"] = int((elbow_ok / len(elbow_checks)) * 10)
            feedback["Swing Control"] = f"Elbow correct in {elbow_ok}/{len(elbow_checks)} frames (first 3s)."

        # Balance (first 3s only)
        balance_checks = [c["balance_ok"] for c in checks_log if "balance_ok" in c]
        if balance_checks:
            balance_ok = sum(balance_checks)
            scores["Balance"] = int((balance_ok / len(balance_checks)) * 10)
            feedback["Balance"] = f"Balanced in {balance_ok}/{len(balance_checks)} frames (first 3s)."

        # Follow-through (last 10% of frames)
        elbow_vals = np.array(metrics_log)[:,0]
        last_n = max(5, int(total_frames * 0.1))
        follow_ok = np.sum(elbow_vals[-last_n:] > 130)
        scores["Follow-through"] = int((follow_ok / last_n) * 10)
        feedback["Follow-through"] = f"Good follow-through in {follow_ok}/{last_n} end frames."

        # Save evaluation
        result = {"scores": scores, "feedback": feedback}
        with open(f"{out_dir}/evaluation.json", "w") as f:
            json.dump(result, f, indent=2)

    # Save all frame values
    with open(f"{out_dir}/all_frame_values.json", "w") as f:
        json.dump(all_frame_values, f, indent=2)

    print("Done. Outputs saved in /output/")

if __name__ == "__main__":
    analyze_video("D:/AIDS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/data/input_video.mp4")
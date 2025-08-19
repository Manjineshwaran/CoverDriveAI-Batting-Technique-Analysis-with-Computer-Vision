import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import json

class AdvancedCricketAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose model with optimized settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher accuracy
            enable_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize tracking variables for phase detection
        self.position_history = deque(maxlen=15)  # Store last 15 frames for velocity calculation
        self.phase_history = deque(maxlen=10)     # Smooth phase transitions
        self.phase_start_frame = 0
        self.current_phase = "Pre-Stance"
        
        # Phase detection parameters
        self.velocity_threshold = 0.02  # Minimum velocity to detect movement
        self.angle_change_threshold = 3  # Degrees per frame
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points with b as vertex."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def calculate_velocity(self, current_pos, prev_pos):
        """Calculate velocity between two positions."""
        if prev_pos is None:
            return 0
        return np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    
    def get_key_landmarks(self, landmarks, frame_shape):
        """Extract key landmarks with frame coordinates."""
        h, w = frame_shape[:2]
        
        try:
            keypoints = {
                'left_shoulder': (landmarks[11].x * w, landmarks[11].y * h),
                'right_shoulder': (landmarks[12].x * w, landmarks[12].y * h),
                'left_elbow': (landmarks[13].x * w, landmarks[13].y * h),
                'right_elbow': (landmarks[14].x * w, landmarks[14].y * h),
                'left_wrist': (landmarks[15].x * w, landmarks[15].y * h),
                'right_wrist': (landmarks[16].x * w, landmarks[16].y * h),
                'left_hip': (landmarks[23].x * w, landmarks[23].y * h),
                'right_hip': (landmarks[24].x * w, landmarks[24].y * h),
                'left_knee': (landmarks[25].x * w, landmarks[25].y * h),
                'right_knee': (landmarks[26].x * w, landmarks[26].y * h),
                'left_ankle': (landmarks[27].x * w, landmarks[27].y * h),
                'right_ankle': (landmarks[28].x * w, landmarks[28].y * h),
                'nose': (landmarks[0].x * w, landmarks[0].y * h),
                'left_index': (landmarks[19].x * w, landmarks[19].y * h),
                'right_index': (landmarks[20].x * w, landmarks[20].y * h)
            }
            return keypoints
        except (IndexError, AttributeError):
            return None
    
    def calculate_batting_metrics(self, keypoints):
        """Calculate comprehensive batting metrics."""
        metrics = {}
        
        try:
            # Knee angles (critical for power generation)
            left_knee_angle = self.calculate_angle(
                keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle']
            )
            right_knee_angle = self.calculate_angle(
                keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle']
            )
            
            # Elbow angles (bat control)
            left_elbow_angle = self.calculate_angle(
                keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist']
            )
            right_elbow_angle = self.calculate_angle(
                keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist']
            )
            
            # Hip angles (rotation and power)
            left_hip_angle = self.calculate_angle(
                keypoints['left_shoulder'], keypoints['left_hip'], keypoints['left_knee']
            )
            right_hip_angle = self.calculate_angle(
                keypoints['right_shoulder'], keypoints['right_hip'], keypoints['right_knee']
            )
            
            # Body alignment metrics
            shoulder_width = abs(keypoints['left_shoulder'][0] - keypoints['right_shoulder'][0])
            hip_width = abs(keypoints['left_hip'][0] - keypoints['right_hip'][0])
            
            # Stance width (important for balance)
            foot_width = abs(keypoints['left_ankle'][0] - keypoints['right_ankle'][0])
            
            # Head position (eye level consistency)
            head_stability = keypoints['nose'][1]
            
            # Bat position approximation (using hand positions)
            bat_height = min(keypoints['left_wrist'][1], keypoints['right_wrist'][1])
            
            # Weight distribution (based on foot positioning)
            weight_center_x = (keypoints['left_ankle'][0] + keypoints['right_ankle'][0]) / 2
            body_center_x = (keypoints['left_hip'][0] + keypoints['right_hip'][0]) / 2
            weight_shift = abs(weight_center_x - body_center_x)
            
            metrics = {
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'avg_knee_angle': (left_knee_angle + right_knee_angle) / 2,
                'left_elbow_angle': left_elbow_angle,
                'right_elbow_angle': right_elbow_angle,
                'left_hip_angle': left_hip_angle,
                'right_hip_angle': right_hip_angle,
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'foot_width': foot_width,
                'head_y': head_stability,
                'bat_height': bat_height,
                'weight_shift': weight_shift,
                'body_center': body_center_x
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
        return metrics
    
    def detect_phase_with_velocity(self, keypoints, metrics, frame_num):
        """Advanced phase detection using velocities and biomechanical analysis."""
        current_data = {
            'keypoints': keypoints,
            'metrics': metrics,
            'frame': frame_num
        }
        
        self.position_history.append(current_data)
        
        if len(self.position_history) < 3:
            return "Pre-Stance", []
        
        # Calculate velocities for key points
        velocities = {}
        if len(self.position_history) >= 2:
            current = self.position_history[-1]['keypoints']
            previous = self.position_history[-2]['keypoints']
            
            for point in ['left_wrist', 'right_wrist', 'left_knee', 'right_knee', 'nose']:
                if point in current and point in previous:
                    velocities[point] = self.calculate_velocity(current[point], previous[point])
        
        # Get current metrics
        knee_angle = metrics.get('avg_knee_angle', 0)
        left_elbow = metrics.get('left_elbow_angle', 0)
        right_elbow = metrics.get('right_elbow_angle', 0)
        bat_height = metrics.get('bat_height', 0)
        weight_shift = metrics.get('weight_shift', 0)
        foot_width = metrics.get('foot_width', 0)
        
        # Hand velocities for bat movement detection
        hand_velocity = max(velocities.get('left_wrist', 0), velocities.get('right_wrist', 0))
        knee_velocity = max(velocities.get('left_knee', 0), velocities.get('right_knee', 0))
        head_velocity = velocities.get('nose', 0)
        
        phase = "Unknown"
        feedback = []
        
        # Enhanced Phase Detection Logic
        
        # 1. PRE-STANCE: Initial position, minimal movement
        if (hand_velocity < 5 and knee_velocity < 3 and 
            knee_angle > 160 and head_velocity < 2):
            phase = "Pre-Stance"
        
        # 2. STANCE: Settled position, knees bent, hands up
        elif (hand_velocity < 8 and knee_velocity < 5 and 
              140 < knee_angle < 165 and bat_height < keypoints['nose'][1]):
            phase = "Stance"
            
            # Stance feedback
            if knee_angle < 145:
                feedback.append("Knees too bent - straighten slightly")
            elif knee_angle > 160:
                feedback.append("Knees too straight - bend more")
            else:
                feedback.append("Good knee bend")
                
            if foot_width < 50:
                feedback.append("Stance too narrow")
            elif foot_width > 120:
                feedback.append("Stance too wide")
        
        # 3. BACKLIFT: Hands moving up and back
        elif (hand_velocity > 8 and bat_height > keypoints['nose'][1] and 
              knee_angle > 150 and right_elbow > 100):
            phase = "Backlift"
            
            if right_elbow < 90:
                feedback.append("Back elbow too low")
            elif right_elbow > 140:
                feedback.append("Back elbow too high")
        
        # 4. STRIDE: Forward movement, weight transfer
        elif (knee_velocity > 3 and weight_shift > 15 and 
              knee_angle > 155 and hand_velocity < 15):
            phase = "Stride"
            
            if weight_shift > 40:
                feedback.append("Too much weight transfer")
            elif head_velocity > 5:
                feedback.append("Head moving too much")
        
        # 5. DOWNSWING: Rapid hand movement downward
        elif (hand_velocity > 15 and bat_height < keypoints['nose'][1] and 
              knee_angle < 160 and right_elbow < 120):
            phase = "Downswing"
            
            if knee_angle > 155:
                feedback.append("Stay low through swing")
            if head_velocity > 8:
                feedback.append("Keep head still")
        
        # 6. IMPACT: Peak hand velocity, optimal position
        elif (hand_velocity > 20 and 130 < knee_angle < 150 and 
              80 < right_elbow < 110 and weight_shift > 10):
            phase = "Impact"
            
            if knee_angle < 135:
                feedback.append("Too low at impact")
            elif knee_angle > 145:
                feedback.append("Too high at impact")
            else:
                feedback.append("Good impact position")
                
            if right_elbow < 85:
                feedback.append("Back elbow too tucked")
            elif right_elbow > 105:
                feedback.append("Back elbow too open")
        
        # 7. FOLLOW-THROUGH: Hands extending, body opening
        elif (hand_velocity > 10 and left_elbow > 140 and 
              right_elbow > 120 and knee_angle > 150):
            phase = "Follow-Through"
            
            if left_elbow < 130:
                feedback.append("Incomplete follow-through")
        
        # 8. RECOVERY: Settling back to balanced position
        elif (hand_velocity < 8 and knee_velocity < 5 and 
              150 < knee_angle < 170 and weight_shift < 20):
            phase = "Recovery"
        
        # Smooth phase transitions
        self.phase_history.append(phase)
        if len(self.phase_history) >= 3:
            # Use majority vote for stability
            recent_phases = list(self.phase_history)[-3:]
            if recent_phases.count(phase) >= 2:
                self.current_phase = phase
            else:
                # Keep previous phase if transition is unclear
                pass
        
        return self.current_phase, feedback
    
    def draw_biomechanical_overlay(self, frame, keypoints, metrics, phase, feedback):
        """Draw detailed biomechanical analysis overlay."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        try:
            # Draw key angles
            if 'left_knee_angle' in metrics:
                # Left knee angle visualization
                cv2.circle(overlay, tuple(map(int, keypoints['left_knee'])), 8, (0, 255, 255), -1)
                cv2.putText(overlay, f"{metrics['left_knee_angle']:.0f}°", 
                           (int(keypoints['left_knee'][0] + 15), int(keypoints['left_knee'][1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if 'right_knee_angle' in metrics:
                # Right knee angle visualization
                cv2.circle(overlay, tuple(map(int, keypoints['right_knee'])), 8, (0, 255, 255), -1)
                cv2.putText(overlay, f"{metrics['right_knee_angle']:.0f}°", 
                           (int(keypoints['right_knee'][0] + 15), int(keypoints['right_knee'][1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw stance width line
            if 'foot_width' in metrics:
                cv2.line(overlay, tuple(map(int, keypoints['left_ankle'])), 
                        tuple(map(int, keypoints['right_ankle'])), (255, 0, 255), 3)
                mid_foot = ((keypoints['left_ankle'][0] + keypoints['right_ankle'][0]) / 2,
                           max(keypoints['left_ankle'][1], keypoints['right_ankle'][1]) + 20)
                cv2.putText(overlay, f"Stance: {metrics['foot_width']:.0f}px", 
                           tuple(map(int, mid_foot)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Draw bat path approximation
            bat_center = ((keypoints['left_wrist'][0] + keypoints['right_wrist'][0]) / 2,
                         (keypoints['left_wrist'][1] + keypoints['right_wrist'][1]) / 2)
            cv2.circle(overlay, tuple(map(int, bat_center)), 6, (0, 165, 255), -1)
            
        except Exception as e:
            print(f"Error drawing overlay: {e}")
        
        return overlay
    
    def get_phase_color(self, phase):
        """Return color coding for different phases."""
        colors = {
            "Pre-Stance": (128, 128, 128),    # Gray
            "Stance": (0, 255, 0),            # Green
            "Backlift": (255, 255, 0),        # Yellow
            "Stride": (255, 165, 0),          # Orange
            "Downswing": (255, 69, 0),        # Red-Orange
            "Impact": (255, 0, 0),            # Red
            "Follow-Through": (147, 20, 255),  # Purple
            "Recovery": (0, 255, 255),        # Cyan
            "Unknown": (64, 64, 64)           # Dark Gray
        }
        return colors.get(phase, (255, 255, 255))
    
    def analyze_technique(self, metrics, phase):
        """Provide detailed technique analysis based on phase."""
        analysis = []
        
        if phase == "Stance":
            knee_angle = metrics.get('avg_knee_angle', 0)
            foot_width = metrics.get('foot_width', 0)
            
            if 145 <= knee_angle <= 155:
                analysis.append("✓ Optimal knee bend")
            elif knee_angle < 145:
                analysis.append("⚠ Knees too bent - reduce crouch")
            else:
                analysis.append("⚠ Knees too straight - lower stance")
                
            if 60 <= foot_width <= 100:
                analysis.append("✓ Good stance width")
            elif foot_width < 60:
                analysis.append("⚠ Stance too narrow")
            else:
                analysis.append("⚠ Stance too wide")
        
        elif phase == "Backlift":
            right_elbow = metrics.get('right_elbow_angle', 0)
            if 100 <= right_elbow <= 130:
                analysis.append("✓ Good backlift height")
            elif right_elbow < 100:
                analysis.append("⚠ Backlift too low")
            else:
                analysis.append("⚠ Backlift too high")
        
        elif phase == "Impact":
            knee_angle = metrics.get('avg_knee_angle', 0)
            right_elbow = metrics.get('right_elbow_angle', 0)
            
            if 135 <= knee_angle <= 145:
                analysis.append("✓ Good impact position")
            elif knee_angle < 135:
                analysis.append("⚠ Too low at impact")
            else:
                analysis.append("⚠ Too high at impact")
                
            if 85 <= right_elbow <= 105:
                analysis.append("✓ Good elbow position")
            elif right_elbow < 85:
                analysis.append("⚠ Elbow too tucked")
            else:
                analysis.append("⚠ Elbow too extended")
        
        elif phase == "Follow-Through":
            left_elbow = metrics.get('left_elbow_angle', 0)
            if left_elbow > 140:
                analysis.append("✓ Complete follow-through")
            else:
                analysis.append("⚠ Incomplete follow-through")
        
        return analysis
    
    def create_analysis_dashboard(self, frame, phase, metrics, feedback, analysis, frame_count, total_frames):
        """Create comprehensive analysis dashboard overlay."""
        h, w = frame.shape[:2]
        
        # Create dashboard area
        dashboard_height = 200
        dashboard = np.zeros((dashboard_height, w, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)  # Dark background
        
        # Phase indicator
        phase_color = self.get_phase_color(phase)
        cv2.rectangle(dashboard, (10, 10), (200, 50), phase_color, -1)
        cv2.putText(dashboard, f"Phase: {phase}", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Progress bar
        progress = frame_count / total_frames
        cv2.rectangle(dashboard, (220, 15), (w-20, 35), (100, 100, 100), 2)
        cv2.rectangle(dashboard, (222, 17), (int(222 + (w-242) * progress), 33), (0, 255, 0), -1)
        cv2.putText(dashboard, f"{frame_count}/{total_frames}", (w-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Key metrics display
        y_pos = 70
        col1_x, col2_x, col3_x = 15, 200, 400
        
        if metrics:
            # Column 1: Knee metrics
            cv2.putText(dashboard, "KNEES:", (col1_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(dashboard, f"L: {metrics.get('left_knee_angle', 0):.0f}°", 
                       (col1_x, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(dashboard, f"R: {metrics.get('right_knee_angle', 0):.0f}°", 
                       (col1_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(dashboard, f"Avg: {metrics.get('avg_knee_angle', 0):.0f}°", 
                       (col1_x, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Column 2: Elbow metrics
            cv2.putText(dashboard, "ELBOWS:", (col2_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(dashboard, f"L: {metrics.get('left_elbow_angle', 0):.0f}°", 
                       (col2_x, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(dashboard, f"R: {metrics.get('right_elbow_angle', 0):.0f}°", 
                       (col2_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Column 3: Stance metrics
            if col3_x < w - 150:
                cv2.putText(dashboard, "STANCE:", (col3_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(dashboard, f"Width: {metrics.get('foot_width', 0):.0f}px", 
                           (col3_x, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(dashboard, f"Weight: {metrics.get('weight_shift', 0):.0f}px", 
                           (col3_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Feedback section
        feedback_y = 150
        cv2.putText(dashboard, "FEEDBACK:", (15, feedback_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Combine feedback and analysis
        all_feedback = feedback + analysis
        for i, fb in enumerate(all_feedback[:3]):  # Show max 3 feedback items
            color = (0, 255, 0) if "✓" in fb or "Good" in fb else (0, 165, 255)
            cv2.putText(dashboard, fb[:50], (15, feedback_y + 20 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Combine dashboard with main frame
        combined = np.vstack([frame, dashboard])
        return combined
    
    def analyze_video(self, video_path, output_path='advanced_cricket_analysis.mp4'):
        """Main video analysis function with enhanced phase detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video with dashboard
        dashboard_height = 200
        output_height = height + dashboard_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))
        
        frame_count = 0
        phase_transitions = []
        
        print(f"Analyzing cricket video: {total_frames} frames at {fps:.1f} FPS")
        print("Detecting phases: Pre-Stance → Stance → Backlift → Stride → Downswing → Impact → Follow-Through → Recovery")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                print(f"Processing: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
            
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            phase = "No Detection"
            feedback = []
            analysis = []
            metrics = {}
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Extract keypoints and calculate metrics
                keypoints = self.get_key_landmarks(results.pose_landmarks.landmark, frame.shape)
                
                if keypoints:
                    metrics = self.calculate_batting_metrics(keypoints)
                    phase, feedback = self.detect_phase_with_velocity(keypoints, metrics, frame_count)
                    analysis = self.analyze_technique(metrics, phase)
                    
                    # Track phase transitions
                    if hasattr(self, 'prev_phase') and phase != self.prev_phase:
                        phase_transitions.append({
                            'frame': frame_count,
                            'from': self.prev_phase,
                            'to': phase,
                            'timestamp': frame_count / fps
                        })
                    self.prev_phase = phase
                    
                    # Draw biomechanical overlay
                    frame = self.draw_biomechanical_overlay(frame, keypoints, metrics, phase, feedback)
            
            # Create final frame with dashboard
            final_frame = self.create_analysis_dashboard(
                frame, phase, metrics, feedback, analysis, frame_count, total_frames
            )
            
            out.write(final_frame)
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print analysis summary
        print(f"\n{'='*60}")
        print("CRICKET BATTING ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Output saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Phase transitions detected: {len(phase_transitions)}")
        
        if phase_transitions:
            print("\nPhase Transition Timeline:")
            for transition in phase_transitions:
                print(f"  {transition['timestamp']:.2f}s: {transition['from']} → {transition['to']}")
        
        return phase_transitions

# Enhanced usage example
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = AdvancedCricketAnalyzer()
    
    # Video file paths
    input_video = "D:/AIDS/cv_job_assignment/AthleteRise_CoverDrive_Analysis/data/input_video.mp4"  # Replace with your video path
    output_video = "advanced_cricket_analysis.mp4"
    
    # Run analysis
    print("Starting Advanced Cricket Batting Analysis...")
    print("Optimized for front-on (bowler's perspective) view")
    print("Features: Automatic phase detection, biomechanical analysis, real-time feedback")
    
    transitions = analyzer.analyze_video(input_video, output_video)
    
    print(f"\nAnalysis complete! Check '{output_video}' for detailed results.")
    
    # Optional: Save phase transition data
    if transitions:
        with open('phase_transitions.json', 'w') as f:
            json.dump(transitions, f, indent=2)
        print("Phase transition data saved to 'phase_transitions.json'")
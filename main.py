import cv2
import json
import logging
import sys
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# Import configuration
from config import (
    OUTPUT_FILES, LOCAL_VIDEO_PATH, POSE_CONFIG, LANDMARK_INDICES,
    METRIC_CONFIG, VISUALIZATION, VIDEO_SETTINGS, LOGGING_CONFIG
)

# Import utilities
from utils.video_utils import (
    initialize_video_capture, initialize_video_writer,
    release_resources, get_video_properties
)
from utils.pose_utils import (
    initialize_pose_estimator, process_frame_for_pose,
    extract_keypoints, draw_skeleton
)
from utils.metric_utils import calculate_metrics, compute_final_scores
from utils.visualization import overlay_metrics, display_frame

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(OUTPUT_FILES['log_file'], mode=LOGGING_CONFIG['filemode']),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('cover_drive_analysis')

def process_video() -> Dict[str, Any]:
    """
    Process video to analyze cricket cover drive technique.
    
    Returns:
        Dict containing analysis results and metrics
        
    Raises:
        Exception: If any error occurs during processing
    """
    # Initialize resources
    cap = None
    writer = None
    pose_estimator = None
    all_metrics = []
    
    try:
        # Step 1: Initialize video capture
        logger.info("Initializing video capture")
        cap, video_props = initialize_video_capture(LOCAL_VIDEO_PATH)
        
        # Step 2: Initialize video writer
        logger.info("Initializing video writer")
        frame_size = (video_props['width'], video_props['height'])
        writer = initialize_video_writer(
            OUTPUT_FILES['annotated_video'],
            min(video_props['fps'], VIDEO_SETTINGS['fps']),
            frame_size,
            fourcc=VIDEO_SETTINGS['fourcc']
        )
        
        # Step 3: Initialize pose estimator
        logger.info("Initializing pose estimator")
        pose_estimator, mp_pose, mp_drawing = initialize_pose_estimator(POSE_CONFIG)
        
        # Step 4: Process each frame
        logger.info("Starting frame processing")
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            logger.debug(f"Processing frame {frame_idx}")
            
            # Initialize metrics for current frame
            metrics = {metric_name: None for metric_name in METRIC_CONFIG}
            annotated_frame = frame.copy()
            
            # Process frame with pose estimation
            results = process_frame_for_pose(pose_estimator, frame)
            
            if results and results.pose_landmarks:
                # Draw skeleton on frame
                annotated_frame = draw_skeleton(
                    frame=annotated_frame,
                    landmarks=results.pose_landmarks,
                    mp_pose=mp_pose,
                    mp_drawing=mp_drawing,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=VISUALIZATION['landmark_color'],
                        thickness=VISUALIZATION['thickness'],
                        circle_radius=VISUALIZATION['circle_radius']
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=VISUALIZATION['connection_color'],
                        thickness=VISUALIZATION['thickness']
                    )
                )
                
                # Extract and process keypoints
                keypoints = extract_keypoints(
                    pose_landmarks=results.pose_landmarks,
                    frame_width=video_props['width'],
                    frame_height=video_props['height'],
                    min_visibility=POSE_CONFIG['min_detection_confidence']
                )
                
                # Calculate metrics for current frame
                metrics = calculate_metrics(
                    keypoints=keypoints,
                    frame_size=frame_size,
                    landmark_indices=LANDMARK_INDICES,
                    metric_config=METRIC_CONFIG
                )
                all_metrics.append(metrics)
            
            # Add metrics overlay to frame
            annotated_frame = overlay_metrics(
                frame=annotated_frame,
                metrics=metrics,
                frame_size=frame_size,
                config=VISUALIZATION
            )
            
            # Write frame to output video
            writer.write(annotated_frame)
            
            # Display progress
            if frame_idx % 50 == 0:
                display_frame("Cover Drive Analysis - Preview", annotated_frame)
        
        # Step 5: Compute final evaluation
        logger.info("Computing final evaluation")
        evaluation = compute_final_scores(
            metrics_history=all_metrics,
            metric_config=METRIC_CONFIG
        )
        
        # Save evaluation results
        try:
            with open(OUTPUT_FILES['evaluation_json'], 'w') as f:
                json.dump(evaluation, f, indent=2)
            logger.info(f"Evaluation saved to {OUTPUT_FILES['evaluation_json']}")
        except Exception as e:
            logger.error(f"Error saving evaluation: {str(e)}")
            raise
        
        return evaluation
        
    except Exception as e:
        logger.critical(f"Analysis failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup resources
        release_resources(cap, writer)
        if pose_estimator:
            pose_estimator.close()
        cv2.destroyAllWindows()
        logger.info("Processing complete")

if __name__ == "__main__":
    """
    Main entry point for the Cover Drive Analysis application.
    """
    try:
        logger.info("=" * 50)
        logger.info("Starting Cover Drive Analysis")
        logger.info("=" * 50)
        
        # Process the video and get evaluation results
        evaluation = process_video()
        
        # Log completion and summary
        logger.info("\n" + "=" * 50)
        logger.info("Analysis completed successfully!")
        logger.info("=" * 50)
        
        # Print summary to console
        if evaluation:
            print("\n=== Analysis Results ===")
            print(f"Overall Score: {evaluation.get('overall_score', 0):.1f}/100")
            print("\n=== Detailed Metrics ===")
            for metric, data in evaluation.get('metrics', {}).items():
                print(f"{metric.replace('_', ' ').title()}: {data.get('value', 0):.1f} "
                      f"(Ideal: {data.get('ideal', 0)}, Score: {data.get('score', 0)}/10)")
            
            print("\n=== Feedback ===")
            for feedback in evaluation.get('feedback', []):
                print(f"- {feedback}")
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.critical(f"Unexpected error occurred: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        logger.info("Application terminated")
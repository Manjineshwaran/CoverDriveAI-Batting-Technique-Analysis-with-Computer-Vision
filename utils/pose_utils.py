"""
Pose estimation utilities for detecting and processing human pose keypoints.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

from exceptions import PoseEstimationError

logger = logging.getLogger('cover_drive_analysis.pose_utils')

# Type aliases
Landmark = Any  # MediaPipe landmark type
PoseResults = Any  # MediaPipe pose results type
PoseEstimator = Any  # MediaPipe pose estimator type


def initialize_pose_estimator(
    config: Dict[str, Any]
) -> Tuple[PoseEstimator, mp.solutions.pose.Pose, Any]:
    """
    Initialize MediaPipe Pose estimator with the given configuration.

    Args:
        config: Configuration dictionary for the pose estimator.
               Expected keys: model_complexity, min_detection_confidence,
               min_tracking_confidence, etc.

    Returns:
        A tuple containing:
        - Initialized pose estimator
        - MediaPipe pose module
        - MediaPipe drawing utilities

    Raises:
        PoseEstimationError: If initialization fails.
    """
    try:
        logger.debug("Initializing MediaPipe Pose estimator")
        
        # Import MediaPipe modules
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        # Set default values for required parameters if not provided
        config.setdefault('model_complexity', 1)
        config.setdefault('min_detection_confidence', 0.5)
        config.setdefault('min_tracking_confidence', 0.5)
        
        # Initialize the pose estimator
        pose_estimator = mp_pose.Pose(**config)
        
        logger.info(
            f"Pose estimator initialized with config: {config}"
        )
        
        return pose_estimator, mp_pose, mp_drawing
        
    except Exception as e:
        error_msg = f"Failed to initialize pose estimator: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise PoseEstimationError(error_msg) from e


def process_frame_for_pose(
    pose_estimator: PoseEstimator,
    frame: np.ndarray,
    frame_count: Optional[int] = None
) -> Optional[PoseResults]:
    """
    Process a single frame for pose estimation.

    Args:
        pose_estimator: Initialized pose estimator.
        frame: Input frame in BGR format.
        frame_count: Optional frame number for logging.

    Returns:
        Pose estimation results or None if processing fails.
    """
    try:
        # Convert BGR to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose_estimator.process(frame_rgb)
        
        # Log frame processing if needed
        if frame_count is not None and frame_count % 100 == 0:
            logger.debug(f"Processed frame {frame_count}")
            
        return results
        
    except Exception as e:
        frame_info = f" at frame {frame_count}" if frame_count is not None else ""
        logger.warning(f"Error processing frame{frame_info}: {str(e)}")
        return None


def extract_keypoints(
    pose_landmarks: Any,
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.5,
    landmark_indices: Optional[List[int]] = None
) -> Dict[int, Tuple[float, float]]:
    """
    Extract and normalize keypoints from pose landmarks.

    Args:
        pose_landmarks: MediaPipe pose landmarks object.
        frame_width: Width of the frame.
        frame_height: Height of the frame.
        min_visibility: Minimum visibility threshold (0.0-1.0).
        landmark_indices: Optional list of landmark indices to extract.
                         If None, all landmarks are extracted.

    Returns:
        Dictionary mapping landmark indices to (x, y) coordinates in frame space.
    """
    keypoints = {}
    
    if pose_landmarks is None or not hasattr(pose_landmarks, 'landmark'):
        return keypoints
    
    # If specific indices are provided, only extract those
    indices_to_extract = landmark_indices if landmark_indices is not None else range(len(pose_landmarks.landmark))
    
    for idx in indices_to_extract:
        if idx < 0 or idx >= len(pose_landmarks.landmark):
            logger.warning(f"Landmark index {idx} out of range")
            continue
            
        landmark = pose_landmarks.landmark[idx]
        
        # Only include landmarks with sufficient visibility
        if landmark.visibility >= min_visibility:
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            keypoints[idx] = (x, y)
    
    return keypoints


def draw_skeleton(
    frame: np.ndarray,
    landmarks: Any,
    mp_pose: Any,
    mp_drawing: Any,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Optional[Any] = None,
    connection_drawing_spec: Optional[Any] = None,
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw the pose skeleton on the input frame.

    Args:
        frame: Input frame in BGR format.
        landmarks: MediaPipe pose landmarks.
        mp_pose: MediaPipe pose module.
        mp_drawing: MediaPipe drawing utilities.
        connections: List of landmark index pairs to connect.
                   If None, uses mp_pose.POSE_CONNECTIONS.
        landmark_drawing_spec: Drawing spec for landmarks.
        connection_drawing_spec: Drawing spec for connections.
        visibility_threshold: Minimum visibility to draw a landmark.

    Returns:
        Frame with pose skeleton drawn.
    """
    if landmarks is None:
        return frame
    
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    # Use default connections if none provided
    if connections is None:
        connections = mp_pose.POSE_CONNECTIONS
    
    # Draw the pose annotations
    mp_drawing.draw_landmarks(
        image=annotated_frame,
        landmark_list=landmarks,
        connections=connections,
        landmark_drawing_spec=landmark_drawing_spec,
        connection_drawing_spec=connection_drawing_spec
    )
    
    return annotated_frame


def calculate_pose_quality(
    landmarks: Any,
    min_landmarks: int = 5,
    min_visibility: float = 0.5
) -> float:
    """
    Calculate a quality score for the detected pose.
    
    Args:
        landmarks: MediaPipe pose landmarks.
        min_landmarks: Minimum number of landmarks needed for a valid pose.
        min_visibility: Minimum visibility threshold for a landmark to be considered.
        
    Returns:
        A quality score between 0.0 (poor) and 1.0 (excellent).
    """
    if landmarks is None or not hasattr(landmarks, 'landmark'):
        return 0.0
    
    # Count visible landmarks
    visible_landmarks = sum(
        1 for lm in landmarks.landmark 
        if lm.visibility >= min_visibility
    )
    
    # If we don't have enough landmarks, quality is 0
    if visible_landmarks < min_landmarks:
        return 0.0
    
    # Calculate quality as a ratio of visible landmarks to total landmarks
    total_landmarks = len(landmarks.landmark)
    return min(1.0, visible_landmarks / total_landmarks * 1.5)  # Cap at 1.0
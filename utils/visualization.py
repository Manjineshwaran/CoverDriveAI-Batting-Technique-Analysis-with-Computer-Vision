"""
Visualization utilities for the cricket cover drive analysis system.

This module provides functions to visualize pose estimation results and metrics
on video frames with configurable styling and layout.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Configure logger
logger = logging.getLogger('cover_drive_analysis.visualization')

# Type aliases
Frame = np.ndarray
Point = Tuple[int, int]
Color = Tuple[int, int, int]
DrawingSpec = Any  # MediaPipe DrawingSpec type


def draw_skeleton(
    frame: Frame,
    landmarks: Any,
    mp_pose: Any,
    mp_drawing: Any,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Optional[DrawingSpec] = None,
    connection_drawing_spec: Optional[DrawingSpec] = None,
    visibility_threshold: float = 0.5
) -> Frame:
    """
    Draw the pose skeleton on the input frame.

    Args:
        frame: Input BGR frame (numpy array).
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
        return frame.copy()
    
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    # Use default connections if none provided
    if connections is None:
        connections = mp_pose.POSE_CONNECTIONS
    
    try:
        # Draw the pose annotations
        mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=landmarks,
            connections=connections,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )
        
        return annotated_frame
        
    except Exception as e:
        logger.error(f"Error drawing skeleton: {str(e)}")
        return frame.copy()


def overlay_metrics(
    frame: Frame,
    metrics: Dict[str, float],
    frame_size: Tuple[int, int],
    config: Dict[str, Any]
) -> Frame:
    """
    Overlay metrics and analysis results on the frame.

    Args:
        frame: Input BGR frame (numpy array).
        metrics: Dictionary of metric names to values.
        frame_size: Tuple of (width, height) of the frame.
        config: Visualization configuration dictionary with styling.

    Returns:
        Frame with metrics overlay.
    """
    if frame is None or frame.size == 0:
        logger.warning("Empty frame provided for overlay")
        return frame
    
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Get styling from config with defaults
    font = config.get('font', cv2.FONT_HERSHEY_SIMPLEX)
    font_scale = config.get('font_scale', 0.7)
    font_thickness = config.get('font_thickness', 2)
    text_color = config.get('text_color', (0, 255, 0))  # BGR format
    bg_color = config.get('bg_color', (0, 0, 0))  # Black background
    bg_opacity = config.get('bg_opacity', 0.6)
    
    # Calculate text positions
    margin = 10
    line_height = int(30 * font_scale)
    x = margin
    y_start = margin + line_height
    
    # Prepare text lines
    lines = []
    
    # Add timestamp if available
    if 'timestamp' in metrics:
        timestamp = metrics['timestamp']
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        lines.append(f"Time: {mins:02d}:{secs:02d}")
    
    # Add metrics with formatting
    for metric_name, value in metrics.items():
        if metric_name == 'timestamp':
            continue
            
        # Format value based on its type
        if value is None:
            formatted_value = "N/A"
        elif isinstance(value, float):
            formatted_value = f"{value:.1f}°" if 'angle' in metric_name else f"{value:.2f}"
        else:
            formatted_value = str(value)
            
        # Format the metric name for display
        display_name = metric_name.replace('_', ' ').title()
        lines.append(f"{display_name}: {formatted_value}")
    
    # Calculate background rectangle size
    if not lines:
        return annotated_frame
    
    # Find the longest line for background width
    max_line_length = max(cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] 
                         for line in lines)
    
    bg_width = max_line_length + 2 * margin
    bg_height = len(lines) * line_height + margin
    
    # Draw semi-transparent background
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (bg_width, bg_height), bg_color, -1)
    cv2.addWeighted(overlay, bg_opacity, annotated_frame, 1 - bg_opacity, 0, annotated_frame)
    
    # Draw text
    for i, line in enumerate(lines):
        y = y_start + i * line_height
        cv2.putText(
            img=annotated_frame,
            text=line,
            org=(x + 5, y),
            fontFace=font,
            fontScale=font_scale,
            color=text_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA
        )
    
    return annotated_frame


def display_frame(
    window_name: str,
    frame: Frame,
    wait_time: int = 1
) -> int:
    """
    Display a frame in a window.

    Args:
        window_name: Name of the window.
        frame: Frame to display.
        wait_time: Time in milliseconds to wait for a key event.

    Returns:
        The key code of the pressed key, or -1 if no key was pressed.
    """
    if frame is None or frame.size == 0:
        logger.warning("Cannot display empty frame")
        return -1
    
    try:
        cv2.imshow(window_name, frame)
        return cv2.waitKey(wait_time) & 0xFF
    except Exception as e:
        logger.error(f"Error displaying frame: {str(e)}")
        return -1


def draw_landmark(
    frame: Frame,
    point: Point,
    color: Color = (0, 255, 0),
    radius: int = 5,
    thickness: int = -1
) -> Frame:
    """
    Draw a single landmark point on the frame.

    Args:
        frame: Input frame.
        point: (x, y) coordinates of the point.
        color: BGR color tuple.
        radius: Radius of the circle.
        thickness: Thickness of the circle outline (-1 for filled).

    Returns:
        Frame with the landmark drawn.
    """
    if frame is None or point is None:
        return frame
        
    try:
        x, y = int(round(point[0])), int(round(point[1]))
        return cv2.circle(frame, (x, y), radius, color, thickness)
    except Exception as e:
        logger.warning(f"Error drawing landmark: {str(e)}")
        return frame


def draw_line(
    frame: Frame,
    pt1: Point,
    pt2: Point,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA
) -> Frame:
    """
    Draw a line between two points on the frame.

    Args:
        frame: Input frame.
        pt1: First point (x1, y1).
        pt2: Second point (x2, y2).
        color: BGR color tuple.
        thickness: Line thickness.
        line_type: Type of the line.

    Returns:
        Frame with the line drawn.
    """
    if frame is None or pt1 is None or pt2 is None:
        return frame
        
    try:
        pt1_int = (int(round(pt1[0])), int(round(pt1[1])))
        pt2_int = (int(round(pt2[0])), int(round(pt2[1])))
        return cv2.line(frame, pt1_int, pt2_int, color, thickness, line_type)
    except Exception as e:
        logger.warning(f"Error drawing line: {str(e)}")
        return frame


def draw_text(
    frame: Frame,
    text: str,
    position: Point,
    font_scale: float = 1.0,
    color: Color = (255, 255, 255),
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    line_type: int = cv2.LINE_AA,
    bg_color: Optional[Color] = None,
    padding: int = 2
) -> Frame:
    """
    Draw text on the frame with optional background.

    Args:
        frame: Input frame.
        text: Text to draw.
        position: (x, y) position of the text.
        font_scale: Font scale factor.
        color: Text color (BGR).
        thickness: Text thickness.
        font: Font type.
        line_type: Line type.
        bg_color: Background color (BGR) or None for no background.
        padding: Padding around the text for background.

    Returns:
        Frame with text drawn.
    """
    if frame is None or not text:
        return frame
        
    try:
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background if specified
        if bg_color is not None:
            cv2.rectangle(
                frame,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                bg_color,
                -1
            )
        
        # Draw text
        cv2.putText(
            img=frame,
            text=text,
            org=(x, y),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=line_type
        )
        
        return frame
        
    except Exception as e:
        logger.warning(f"Error drawing text: {str(e)}")
        return frame
"""
Metric calculation utilities for analyzing cricket cover drive technique.

This module provides functions to calculate various metrics related to cricket
cover drive technique using pose estimation keypoints.
"""

import math
import logging
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from exceptions import MetricCalculationError

logger = logging.getLogger('cover_drive_analysis.metric_utils')

# Type aliases
Point2D = Tuple[float, float]
Keypoints = Dict[int, Point2D]
MetricResult = Dict[str, Union[float, Dict[str, float]]]


def calculate_metrics(
    keypoints: Keypoints,
    frame_size: Tuple[int, int],
    landmark_indices: Dict[str, int],
    metric_config: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate all metrics for a single frame based on detected keypoints.

    Args:
        keypoints: Dictionary mapping landmark indices to (x, y) coordinates.
        frame_size: Tuple of (width, height) of the frame.
        landmark_indices: Dictionary mapping landmark names to their indices.
        metric_config: Configuration for each metric including thresholds and weights.

    Returns:
        Dictionary mapping metric names to their calculated values.
    """
    metrics = {}
    
    # Check if we have enough keypoints to proceed
    if not keypoints or len(keypoints) < 3:  # At least 3 points needed for any meaningful calculation
        logger.warning("Insufficient keypoints for metric calculation")
        return {}
    
    # Calculate each metric defined in the config
    for metric_name, config in metric_config.items():
        try:
            # Skip disabled metrics
            if not config.get('enabled', True):
                continue
                
            # Get the appropriate calculation function
            func_name = f"_calculate_{metric_name}"
            if not hasattr(sys.modules[__name__], func_name):
                logger.warning(f"No calculation function for metric: {metric_name}")
                continue
                
            calc_func = getattr(sys.modules[__name__], func_name)
            
            # Calculate the metric
            metric_value = calc_func(keypoints, landmark_indices, frame_size, config)
            metrics[metric_name] = metric_value
            
        except Exception as e:
            logger.warning(f"Error calculating metric '{metric_name}': {str(e)}")
            metrics[metric_name] = None
    
    return metrics


def compute_final_scores(
    metrics_history: List[Dict[str, float]],
    metric_config: Dict[str, Dict[str, Any]],
    min_frames: int = 10
) -> Dict[str, Any]:
    """
    Compute final scores and feedback based on metrics history.

    Args:
        metrics_history: List of metric dictionaries from each frame.
        metric_config: Configuration for each metric including thresholds and weights.
        min_frames: Minimum number of frames required for valid analysis.

    Returns:
        Dictionary containing:
        - overall_score: Weighted average of all metric scores (0-100)
        - metrics: Detailed scores and values for each metric
        - feedback: List of feedback strings for the athlete
    """
    if not metrics_history or len(metrics_history) < min_frames:
        raise MetricCalculationError("Insufficient frames for analysis")
    
    # Initialize results
    results = {
        'overall_score': 0.0,
        'metrics': {},
        'feedback': []
    }
    
    total_weight = 0.0
    weighted_score_sum = 0.0
    
    # Process each metric
    for metric_name, config in metric_config.items():
        if not config.get('enabled', True):
            continue
            
        # Extract values for this metric across all frames (ignoring None values)
        values = [m.get(metric_name) for m in metrics_history if m.get(metric_name) is not None]
        
        if not values:
            logger.warning(f"No valid values for metric: {metric_name}")
            continue
            
        # Calculate statistics
        avg_value = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)
        
        # Calculate score based on thresholds
        score = _calculate_metric_score(avg_value, config)
        
        # Store metric results
        results['metrics'][metric_name] = {
            'value': avg_value,
            'min': min_value,
            'max': max_value,
            'score': score,
            'weight': config.get('weight', 1.0),
            'ideal': config.get('ideal_value', 0.0),
            'tolerance': config.get('tolerance', 0.0)
        }
        
        # Add to overall score calculation
        weight = config.get('weight', 1.0)
        weighted_score_sum += score * weight
        total_weight += weight
        
        # Generate feedback
        feedback = _generate_metric_feedback(metric_name, avg_value, score, config)
        if feedback:
            results['feedback'].append(feedback)
    
    # Calculate overall weighted score (0-100)
    if total_weight > 0:
        results['overall_score'] = min(100.0, max(0.0, weighted_score_sum / total_weight))
    
    # Add overall feedback based on score
    overall_feedback = _generate_overall_feedback(results['overall_score'])
    if overall_feedback:
        results['feedback'].insert(0, overall_feedback)
    
    return results


def _calculate_metric_score(value: float, config: Dict[str, Any]) -> float:
    """
    Calculate a score (0-100) for a metric based on its value and configuration.
    
    Args:
        value: The calculated metric value.
        config: Metric configuration with thresholds and scoring parameters.
        
    Returns:
        Score between 0 and 100.
    """
    ideal = config.get('ideal_value', 0.0)
    tolerance = config.get('tolerance', 0.0)
    
    # If we have a range (min/max) instead of a single ideal value
    if 'min_value' in config and 'max_value' in config:
        min_val = config['min_value']
        max_val = config['max_value']
        
        if value < min_val or value > max_val:
            return 0.0  # Outside acceptable range
            
        # Score based on distance from the middle of the range
        mid = (min_val + max_val) / 2
        distance = abs(value - mid)
        max_distance = (max_val - min_val) / 2
        
        # Linear score from 0 to 100 based on distance from middle
        return max(0.0, 100.0 * (1.0 - (distance / max_distance)))
    
    # Single ideal value with tolerance
    else:
        distance = abs(value - ideal)
        
        # If within tolerance, full score
        if distance <= tolerance:
            return 100.0
            
        # Linear decrease from 100 to 0 as distance increases
        max_distance = config.get('max_deviation', tolerance * 2)
        score = max(0.0, 100.0 * (1.0 - (distance - tolerance) / (max_distance - tolerance)))
        return score


def _generate_metric_feedback(
    metric_name: str,
    value: float,
    score: float,
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Generate feedback for a specific metric based on its value and score.
    
    Args:
        metric_name: Name of the metric.
        value: Calculated value of the metric.
        score: Score (0-100) for the metric.
        config: Metric configuration.
        
    Returns:
        Feedback string or None if no feedback needed.
    """
    # Skip if score is good
    if score >= 80.0:
        return None
    
    # Get feedback template from config or use default
    feedback_template = config.get(
        'feedback_template',
        "{metric}: {value:.1f} (Ideal: {ideal}±{tolerance})"
    )
    
    # Format the feedback
    feedback = feedback_template.format(
        metric=metric_name.replace('_', ' ').title(),
        value=value,
        ideal=config.get('ideal_value', '?'),
        tolerance=config.get('tolerance', '?')
    )
    
    return feedback


def _generate_overall_feedback(score: float) -> str:
    """
    Generate overall feedback based on the final score.
    
    Args:
        score: Overall score (0-100).
        
    Returns:
        Feedback string.
    """
    if score >= 90:
        return "Excellent technique! Your cover drive is well-executed with great form."
    elif score >= 75:
        return "Good technique overall. A few minor adjustments could improve your form."
    elif score >= 60:
        return "Fair technique. Focus on the key areas highlighted in the feedback."
    else:
        return "Needs improvement. Please review the feedback and practice the fundamentals."


# ===== Individual Metric Calculation Functions =====

def _calculate_elbow_angle(
    keypoints: Keypoints,
    landmark_indices: Dict[str, int],
    frame_size: Tuple[int, int],
    config: Dict[str, Any]
) -> float:
    """
    Calculate the angle at the elbow joint.
    
    Args:
        keypoints: Dictionary of landmark indices to (x, y) coordinates.
        landmark_indices: Dictionary mapping landmark names to indices.
        frame_size: Tuple of (width, height) of the frame.
        config: Configuration for this metric.
        
    Returns:
        Angle in degrees.
    """
    # Get required landmark indices
    shoulder = landmark_indices.get('LEFT_SHOULDER' if config.get('is_left_handed', False) else 'RIGHT_SHOULDER')
    elbow = landmark_indices.get('LEFT_ELBOW' if config.get('is_left_handed', False) else 'RIGHT_ELBOW')
    wrist = landmark_indices.get('LEFT_WRIST' if config.get('is_left_handed', False) else 'RIGHT_WRIST')
    
    # Check if we have all required keypoints
    if not all(k in keypoints for k in [shoulder, elbow, wrist] if k is not None):
        return None
    
    # Calculate vectors
    vec1 = (keypoints[shoulder][0] - keypoints[elbow][0], 
            keypoints[shoulder][1] - keypoints[elbow][1])
    vec2 = (keypoints[wrist][0] - keypoints[elbow][0], 
            keypoints[wrist][1] - keypoints[elbow][1])
    
    # Calculate angle in degrees
    angle = _calculate_angle_between_vectors(vec1, vec2)
    return angle


def _calculate_spine_lean(
    keypoints: Keypoints,
    landmark_indices: Dict[str, int],
    frame_size: Tuple[int, int],
    config: Dict[str, Any]
) -> float:
    """
    Calculate the forward lean of the spine from vertical.
    
    Args:
        keypoints: Dictionary of landmark indices to (x, y) coordinates.
        landmark_indices: Dictionary mapping landmark names to indices.
        frame_size: Tuple of (width, height) of the frame.
        config: Configuration for this metric.
        
    Returns:
        Angle in degrees from vertical.
    """
    # Get required landmark indices
    shoulder = landmark_indices.get('LEFT_SHOULDER'), landmark_indices.get('RIGHT_SHOULDER')
    hip = landmark_indices.get('LEFT_HIP'), landmark_indices.get('RIGHT_HIP')
    
    # Check if we have all required keypoints
    if not all(k in keypoints for k in [shoulder[0], shoulder[1], hip[0], hip[1]] if k is not None):
        return None
    
    # Calculate midpoints
    shoulder_mid = (
        (keypoints[shoulder[0]][0] + keypoints[shoulder[1]][0]) / 2,
        (keypoints[shoulder[0]][1] + keypoints[shoulder[1]][1]) / 2
    )
    hip_mid = (
        (keypoints[hip[0]][0] + keypoints[hip[1]][0]) / 2,
        (keypoints[hip[0]][1] + keypoints[hip[1]][1]) / 2
    )
    
    # Calculate angle from vertical
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    
    # Avoid division by zero
    if dx == 0:
        return 90.0
    
    # Calculate angle in degrees (0 = vertical, 90 = horizontal)
    angle_rad = math.atan2(abs(dx), abs(dy))
    angle_deg = math.degrees(angle_rad)
    
    # Determine direction of lean (forward/backward)
    if dx > 0:  # Leaning to the right (for right-handed batter)
        angle_deg = -angle_deg
    
    return angle_deg


def _calculate_head_knee_distance(
    keypoints: Keypoints,
    landmark_indices: Dict[str, int],
    frame_size: Tuple[int, int],
    config: Dict[str, Any]
) -> float:
    """
    Calculate the horizontal distance between head and front knee.
    
    Args:
        keypoints: Dictionary of landmark indices to (x, y) coordinates.
        landmark_indices: Dictionary mapping landmark names to indices.
        frame_size: Tuple of (width, height) of the frame.
        config: Configuration for this metric.
        
    Returns:
        Normalized distance (0-1).
    """
    # Get required landmark indices
    nose = landmark_indices.get('NOSE')
    front_knee = landmark_indices.get('LEFT_KNEE' if config.get('is_left_handed', False) else 'RIGHT_KNEE')
    
    # Check if we have all required keypoints
    if not all(k in keypoints for k in [nose, front_knee] if k is not None):
        return None
    
    # Calculate horizontal distance
    distance = abs(keypoints[nose][0] - keypoints[front_knee][0])
    
    # Normalize by frame width
    normalized_distance = distance / frame_size[0]
    
    return normalized_distance


def _calculate_foot_angle(
    keypoints: Keypoints,
    landmark_indices: Dict[str, int],
    frame_size: Tuple[int, int],
    config: Dict[str, Any]
) -> float:
    """
    Calculate the angle of the front foot relative to the ground.
    
    Args:
        keypoints: Dictionary of landmark indices to (x, y) coordinates.
        landmark_indices: Dictionary mapping landmark names to indices.
        frame_size: Tuple of (width, height) of the frame.
        config: Configuration for this metric.
        
    Returns:
        Angle in degrees.
    """
    # Get required landmark indices
    ankle = landmark_indices.get('LEFT_ANKLE' if config.get('is_left_handed', False) else 'RIGHT_ANKLE')
    toe = landmark_indices.get('LEFT_FOOT_INDEX' if config.get('is_left_handed', False) else 'RIGHT_FOOT_INDEX')
    
    # Check if we have all required keypoints
    if not all(k in keypoints for k in [ankle, toe] if k is not None):
        return None
    
    # Calculate vector from ankle to toe
    dx = keypoints[toe][0] - keypoints[ankle][0]
    dy = keypoints[toe][1] - keypoints[ankle][1]
    
    # Calculate angle from horizontal (in degrees)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to 0-180 range
    if angle_deg < 0:
        angle_deg += 180
    
    return angle_deg


# ===== Helper Functions =====

def _calculate_angle_between_vectors(
    vec1: Tuple[float, float], 
    vec2: Tuple[float, float]
) -> float:
    """
    Calculate the angle in degrees between two 2D vectors.
    
    Args:
        vec1: First vector as (x, y).
        vec2: Second vector as (x, y).
        
    Returns:
        Angle in degrees between 0 and 180.
    """
    # Calculate dot product
    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    
    # Calculate magnitudes
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate angle in radians, then convert to degrees
    cos_angle = dot / (mag1 * mag2)
    # Clamp to valid range to avoid numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    
    return math.degrees(angle_rad)


def _distance_between_points(
    point1: Tuple[float, float], 
    point2: Tuple[float, float]
) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        point1: First point as (x, y).
        point2: Second point as (x, y).
        
    Returns:
        Euclidean distance between the points.
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
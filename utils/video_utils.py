"""
Video utility functions for handling video input/output operations.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from exceptions import VideoProcessingError

logger = logging.getLogger('cover_drive_analysis.video_utils')

def initialize_video_capture(
    video_path: Union[str, Path],
    api_preference: Optional[int] = None
) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
    """
    Initialize video capture and return video properties.

    Args:
        video_path: Path to the input video file.
        api_preference: Preferred API backend (e.g., cv2.CAP_FFMPEG).

    Returns:
        Tuple of (video_capture, video_properties).
        video_properties contains:
        - fps: Frames per second
        - width: Frame width
        - height: Frame height
        - frame_count: Total number of frames
        - codec: Video codec
        - format: Video format

    Raises:
        VideoProcessingError: If video cannot be opened or read.
    """
    try:
        video_path = str(video_path)
        logger.debug(f"Opening video: {video_path}")
        
        # Try to open with preferred API if specified
        if api_preference is not None:
            cap = cv2.VideoCapture(video_path, api_preference)
        else:
            cap = cv2.VideoCapture(video_path)
            
        if not cap.isOpened():
            # Try with default backend if preferred API fails
            if api_preference is not None:
                logger.warning(f"Failed with API {api_preference}, trying default backend")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError(f"Could not open video with default backend: {video_path}")
            else:
                raise IOError(f"Could not open video: {video_path}")
        
        # Get video properties
        props = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'format': get_video_format(video_path)
        }
        
        # Validate properties
        if props['fps'] <= 0:
            logger.warning(f"Invalid FPS ({props['fps']}), defaulting to 30")
            props['fps'] = 30.0
            
        if props['width'] <= 0 or props['height'] <= 0:
            raise VideoProcessingError("Invalid video dimensions")
        
        logger.info(
            f"Video initialized: {props['width']}x{props['height']} "
            f"at {props['fps']:.1f} FPS, {props['frame_count']} frames, "
            f"format: {props['format']}"
        )
        
        return cap, props
        
    except Exception as e:
        logger.error(f"Error initializing video capture: {str(e)}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        raise VideoProcessingError(f"Video initialization failed: {str(e)}")

def initialize_video_writer(
    output_path: Union[str, Path],
    fps: float,
    frame_size: Tuple[int, int],
    fourcc: str = 'mp4v',
    is_color: bool = True,
    api_preference: Optional[int] = None
) -> cv2.VideoWriter:
    """
    Initialize a video writer for saving processed frames.

    Args:
        output_path: Path to save the output video.
        fps: Frames per second for the output video.
        frame_size: Tuple of (width, height) for the output frames.
        fourcc: FourCC code for the video codec (default: 'mp4v').
        is_color: Whether the video is color (True) or grayscale (False).
        api_preference: Preferred API backend.

    Returns:
        Initialized VideoWriter object.

    Raises:
        VideoProcessingError: If the video writer cannot be initialized.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert FourCC code to integer
        if isinstance(fourcc, str) and len(fourcc) == 4:
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        
        # Initialize video writer with optional API preference
        writer_params = [
            str(output_path),
            fourcc,
            fps,
            (int(frame_size[0]), int(frame_size[1])),
            is_color
        ]
        
        if api_preference is not None:
            writer = cv2.VideoWriter(*writer_params, api_preference)
        else:
            writer = cv2.VideoWriter(*writer_params)
        
        if not writer.isOpened():
            raise IOError(f"Could not initialize video writer at {output_path}")
        
        logger.info(
            f"Video writer initialized: {frame_size[0]}x{frame_size[1]} "
            f"at {fps} FPS, codec: {fourcc_to_str(fourcc)}"
        )
        
        return writer
        
    except Exception as e:
        logger.error(f"Error initializing video writer: {str(e)}")
        raise VideoProcessingError(f"Video writer initialization failed: {str(e)}")

def release_resources(*resources) -> None:
    """
    Safely release multiple OpenCV resources.
    
    Args:
        *resources: Variable number of resources to release.
                   Can be VideoCapture, VideoWriter, or other OpenCV objects.
    """
    for resource in resources:
        try:
            if resource is not None:
                if hasattr(resource, 'isOpened') and callable(resource.isOpened):
                    if resource.isOpened():
                        resource.release()
                        logger.debug(f"Released resource: {type(resource).__name__}")
                elif hasattr(resource, 'release') and callable(resource.release):
                    resource.release()
                    logger.debug(f"Released resource: {type(resource).__name__}")
        except Exception as e:
            logger.warning(f"Error releasing resource: {str(e)}")

def get_video_properties(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get video properties without loading the entire video.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Dictionary containing video properties.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
            
        props = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'format': get_video_format(video_path)
        }
        cap.release()
        return props
        
    except Exception as e:
        logger.error(f"Error getting video properties: {str(e)}")
        raise VideoProcessingError(f"Failed to get video properties: {str(e)}")

def get_video_format(video_path: Union[str, Path]) -> str:
    """
    Get the format/container of a video file.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        String representing the video format (e.g., 'mp4', 'avi').
    """
    video_path = str(video_path)
    return Path(video_path).suffix.lower().lstrip('.')

def fourcc_to_str(fourcc: int) -> str:
    """
    Convert FourCC code to string representation.
    
    Args:
        fourcc: FourCC code as integer.
        
    Returns:
        String representation of the FourCC code.
    """
    return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
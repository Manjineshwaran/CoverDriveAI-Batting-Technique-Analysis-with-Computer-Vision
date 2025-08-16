import streamlit as st
import os
import json
import time
import cv2
import tempfile
from pathlib import Path
from typing import Dict, Any
import subprocess

# Import your existing modules
from main import process_video
from config import (
    OUTPUT_FILES, LOCAL_VIDEO_PATH, POSE_CONFIG, LANDMARK_INDICES,
    METRIC_CONFIG, VISUALIZATION, VIDEO_SETTINGS, LOGGING_CONFIG
)

# Set page config
st.set_page_config(
    page_title="Cricket Cover Drive Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
        }
        .metric-box {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f0f2f6;
        }
        .good-score {
            color: #2ecc71;
            font-weight: bold;
        }
        .fair-score {
            color: #f39c12;
            font-weight: bold;
        }
        .poor-score {
            color: #e74c3c;
            font-weight: bold;
        }
        .video-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .stButton button {
            width: 100%;
        }
        .stDownloadButton button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

def get_video_duration(video_path: str) -> float:
    """Get duration of video in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return float(result.stdout)
    except:
        return 0

def display_metric_score(name: str, value: float, ideal: float, score: float):
    """Display a metric with colored score based on performance."""
    # Format the metric name for display
    display_name = name.replace('_', ' ').title()
    
    # Determine score class for styling
    if score >= 80:
        score_class = "good-score"
        feedback = "Excellent"
    elif score >= 60:
        score_class = "fair-score"
        feedback = "Good"
    else:
        score_class = "poor-score"
        feedback = "Needs Improvement"
    
    # Create the metric box
    st.markdown(f"""
        <div class="metric-box">
            <h4>{display_name}</h4>
            <p><b>Value:</b> {value:.1f} (Ideal: {ideal:.1f})</p>
            <p><b>Score:</b> <span class="{score_class}">{score:.1f}/100</span> - {feedback}</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    # App title and description
    st.title("🏏 Cricket Cover Drive Analysis")
    st.markdown("""
        Analyze your cricket cover drive technique using computer vision and pose estimation. 
        Upload a video of your shot to get detailed feedback on your form.
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Video upload option
        uploaded_file = st.file_uploader(
            "Upload your cover drive video",
            type=["mp4", "avi", "mov"],
            help="Upload a short video (10-30 seconds) of your cover drive shot"
        )
        
        # Model complexity selection
        model_complexity = st.select_slider(
            "Pose Model Complexity",
            options=[0, 1, 2],
            value=POSE_CONFIG["model_complexity"],
            help="Higher complexity is more accurate but slower"
        )
        
        # Confidence thresholds
        detection_confidence = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=POSE_CONFIG["min_detection_confidence"],
            step=0.05,
            help="Minimum confidence for pose detection"
        )
        
        tracking_confidence = st.slider(
            "Tracking Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=POSE_CONFIG["min_tracking_confidence"],
            step=0.05,
            help="Minimum confidence for pose tracking"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
            This tool analyzes cricket batting technique using MediaPipe for pose estimation.
            It evaluates key aspects of the cover drive including:
            - Elbow position
            - Spine lean
            - Head position
            - Foot alignment
        """)
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Update config with user settings
        POSE_CONFIG["model_complexity"] = model_complexity
        POSE_CONFIG["min_detection_confidence"] = detection_confidence
        POSE_CONFIG["min_tracking_confidence"] = tracking_confidence
        
        # Process the video
        with st.spinner("Analyzing your cover drive..."):
            try:
                # Set the video path in config to the uploaded file
                LOCAL_VIDEO_PATH = temp_video_path
                
                # Process the video
                evaluation = process_video()
                
                if evaluation:
                    # Display results
                    st.success("Analysis complete!")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Results Summary", "Detailed Metrics", "Raw Data"])
                    
                    with tab1:
                        # Overall score
                        overall_score = evaluation.get('overall_score', 0)
                        
                        if overall_score >= 80:
                            st.success(f"Overall Score: {overall_score:.1f}/100 - Excellent technique!")
                        elif overall_score >= 60:
                            st.warning(f"Overall Score: {overall_score:.1f}/100 - Good technique with room for improvement")
                        else:
                            st.error(f"Overall Score: {overall_score:.1f}/100 - Needs significant improvement")
                        
                        # Display annotated video
                        st.subheader("Annotated Video")
                        if os.path.exists(OUTPUT_FILES["annotated_video"]):
                            # Get video duration for better display
                            duration = get_video_duration(OUTPUT_FILES["annotated_video"])
                            
                            # Display video with controls
                            video_file = open(OUTPUT_FILES["annotated_video"], 'rb')
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                            
                            # Download button
                            st.download_button(
                                label="Download Annotated Video",
                                data=video_bytes,
                                file_name="cover_drive_analysis.mp4",
                                mime="video/mp4"
                            )
                        else:
                            st.warning("Annotated video not found")
                        
                        # Key feedback
                        st.subheader("Key Feedback")
                        for feedback in evaluation.get('feedback', []):
                            st.write(f"- {feedback}")
                    
                    with tab2:
                        # Detailed metrics
                        st.subheader("Detailed Metrics Analysis")
                        for metric_name, data in evaluation.get('metrics', {}).items():
                            display_metric_score(
                                metric_name,
                                data.get('value', 0),
                                data.get('ideal', 0),
                                data.get('score', 0)
                            )
                    
                    with tab3:
                        # Raw JSON data
                        st.subheader("Raw Analysis Data")
                        st.json(evaluation)
                        
                        # Download JSON button
                        json_str = json.dumps(evaluation, indent=2)
                        st.download_button(
                            label="Download JSON Data",
                            data=json_str,
                            file_name="cover_drive_analysis.json",
                            mime="application/json"
                        )
                
                else:
                    st.error("Analysis failed to produce results")
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)
            
            finally:
                # Clean up temp files
                try:
                    os.remove(temp_video_path)
                except:
                    pass
    else:
        # Show demo/instructions when no video is uploaded
        st.info("Please upload a video to begin analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("How to Use")
            st.markdown("""
                1. Upload a clear video of your cover drive (10-30 seconds)
                2. Adjust analysis settings if needed
                3. Click 'Analyze' to process your video
                4. View your results and feedback
                
                For best results:
                - Film from a side angle
                - Ensure good lighting
                - Wear clothing that contrasts with the background
            """)
        
        with col2:
            st.subheader("Example Video")
            st.video("https://youtu.be/vSX3IRxGnNY")  # Your example video URL

if __name__ == "__main__":
    main()
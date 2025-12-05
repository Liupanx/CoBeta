"""
Gradio frontend for Beta pose tracking.

Run with: python -m src.beta.gradio_app
or: python src/beta/gradio_app.py
"""
import sys
from pathlib import Path
import tempfile
import cv2

# Handle imports
try:
    from . import Beta
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from beta import Beta

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    sys.exit(1)


def process_video(video_file, track_points_str):
    """
    Process uploaded video and return the output video with trajectories.
    
    Args:
        video_file: Uploaded video file (Gradio file object)
        track_points_str: Comma-separated string of track points (e.g., "hip_mid,left_hand,right_hand")
    
    Returns:
        Path to output video file and status message
    """
    if video_file is None:
        return None, "Please upload a video file."
    
    # Get video file path (Gradio Video component returns a string path)
    if isinstance(video_file, str):
        video_path = video_file
    elif hasattr(video_file, 'name'):
        video_path = video_file.name
    else:
        video_path = str(video_file)
    
    # Parse track points
    if track_points_str and track_points_str.strip():
        track_points = [p.strip() for p in track_points_str.split(",") if p.strip()]
    else:
        track_points = None  # Use defaults
    
    # Create Beta instance
    beta = Beta(track_points=track_points)
    
    # Create temporary output file
    output_path = Path(tempfile.gettempdir()) / f"output_{Path(video_path).stem}.mp4"
    
    # Open video for reading
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None, "Error: Could not open video file."
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if unknown
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer (write directly, don't store frames in memory)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        return None, "Error: Could not create output video file."
    
    # Process video frame by frame
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    try:
        from .utils.pose_helper import get_track_point_coords
        from .utils.draw_helper import draw_trajectory
    except ImportError:
        from utils.pose_helper import get_track_point_coords
        from utils.draw_helper import draw_trajectory
    
    try:
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Rotate if needed (remove this line if your videos don't need rotation)
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                # MediaPipe expects RGB images
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                # Draw landmarks and trajectories
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                    
                    # Extract and store trajectory points
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Extract and track each body part
                    for track_point in beta.track_points:
                        result = get_track_point_coords(
                            track_point,
                            results.pose_landmarks.landmark,
                            frame_width,
                            frame_height
                        )
                        
                        if result is not None:
                            mid_point, _mid_point_3d = result
                            
                            # Add this point to our trajectory
                            beta.trajectories[track_point].append(mid_point)
                            
                            # Get color for this track point
                            color = beta.track_point_colors.get(track_point, (255, 255, 255))
                            
                            # Draw the track point
                            cv2.circle(frame, mid_point, 8, color, -1)
                            
                            # Draw the trajectory path
                            trajectory = beta.trajectories[track_point]
                            if len(trajectory) > 1:
                                draw_trajectory(frame, trajectory, color, thickness=2)
                
                # Write frame directly to output video (don't store in memory)
                out.write(frame)
                frame_count += 1
                
                # Progress update every 30 frames
                if frame_count % 30 == 0 and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    except Exception as e:
        cap.release()
        out.release()
        return None, f"Error during processing: {str(e)}"
    finally:
        # Clean up resources
        cap.release()
        out.release()
    
    if frame_count == 0:
        return None, "Error: No frames processed."
    
    return str(output_path), f"Successfully processed {frame_count} frames! Tracked: {', '.join(beta.track_points)}"


# Create Gradio interface
def create_interface():
    """Create and return Gradio interface."""
    
    # Available track points
    available_points = [
        "hip_mid",
        "upper_body_center", 
        "head",
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot"
    ]
    
    with gr.Blocks(title="Beta Pose Tracker") as demo:
        gr.Markdown("# 🎯 Beta Pose Tracker")
        gr.Markdown("Upload a video to analyze pose trajectories using MediaPipe.")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="Upload Video"
                )
                
                track_points_input = gr.Textbox(
                    label="Track Points (comma-separated)",
                    placeholder="hip_mid,left_hand,right_hand",
                    value="hip_mid,left_hand,right_hand",
                    info=f"Available: {', '.join(available_points)}"
                )
                
                analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Output Video with Trajectories")
                status_output = gr.Textbox(label="Status", interactive=False)
        
        # Examples
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[],
            inputs=[video_input, track_points_input],
            label="Example videos (add your own)"
        )
        
        # Event handlers
        analyze_btn.click(
            fn=process_video,
            inputs=[video_input, track_points_input],
            outputs=[video_output, status_output],
            show_progress=True
        )
        
        gr.Markdown("""
        ### Instructions:
        1. Upload a video file (MP4, AVI, MOV, etc.)
        2. Optionally specify which body parts to track (default: hip_mid, left_hand, right_hand)
        3. Click "Analyze" to process the video
        4. Download the output video with trajectory overlays
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


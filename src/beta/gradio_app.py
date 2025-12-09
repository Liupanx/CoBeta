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


def process_video(video_file, track_points_list, max_resolution):
    """
    Process uploaded video and return the output video with trajectories.
    
    Args:
        video_file: Uploaded video file (Gradio file object)
        track_points_list: List of selected track points (e.g., ["hip_mid", "left_hand", "right_hand"])
        max_resolution: Maximum resolution (e.g., "1920x1080", "1280x720", "854x480")
    
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
    
    # Use selected track points, or defaults if none selected
    if track_points_list and len(track_points_list) > 0:
        track_points = track_points_list
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
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use original dimensions (no rotation needed - video is already in correct orientation)
    # Parse max resolution and calculate target dimensions
    if max_resolution and max_resolution.strip() and 'x' in max_resolution:
        max_w, max_h = map(int, max_resolution.split('x'))
        # Calculate scaling to fit within max resolution while maintaining aspect ratio
        scale = min(max_w / original_width, max_h / original_height, 1.0)
        output_width = int(original_width * scale)
        output_height = int(original_height * scale)
    else:
        # No resizing (Original selected or empty)
        output_width = original_width
        output_height = original_height
        scale = 1.0
    
    resize_needed = scale < 1.0
    if resize_needed:
        print(f"Resizing video from {original_width}x{original_height} to {output_width}x{output_height} (scale: {scale:.2f})")
    else:
        print(f"Processing video: {original_width}x{original_height}")
    
    # Initialize video writer (write directly, don't store frames in memory)
    # Use rotated dimensions for output video
    # Try browser-compatible codecs first (H.264), fallback to others if needed
    codecs_to_try = [
        ('avc1', 'H.264/AVC - most browser compatible'),
        ('H264', 'H.264 alternative'),
        ('mp4v', 'MPEG-4 fallback'),
    ]
    
    out = None
    for codec, description in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
        if out.isOpened():
            print(f"Using codec: {codec} ({description})")
            break
    
    if not out or not out.isOpened():
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
                
                # Resize frame if needed (before processing)
                if resize_needed:
                    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                
                # Get current frame dimensions (after rotation and resize)
                frame_height, frame_width = frame.shape[:2]
                
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
    
    size_info = f" ({output_width}x{output_height})"
    resize_info = f" Resized from {original_width}x{original_height}." if resize_needed else ""
    
    return str(output_path), f"Successfully processed {frame_count} frames{size_info}!{resize_info} Tracked: {', '.join(beta.track_points)}"


# Placeholder function for route segmentation
def process_route_segmentation(video_file):
    """Placeholder for route segmentation functionality."""
    if video_file is None:
        return None, "Please upload a video file for route segmentation."
    return None, "Route segmentation feature coming soon!"


def update_route_video_source(source_choice, trajectory_video):
    """Update route video input based on user's choice."""
    if source_choice == "use_same":
        # Use the video from trajectory tracking tab
        return trajectory_video
    else:
        # Return None to allow new upload
        return None


# Create Gradio interface
def create_interface():
    """Create and return Gradio interface with tabs and custom theme."""
    
    # Load CSS from external file
    css_path = Path(__file__).parent / "utils" / "styles.css"
    custom_css = ""
    try:
        if css_path.exists():
            with open(css_path, 'r', encoding='utf-8') as f:
                custom_css = f.read()
        else:
            print(f"Warning: CSS file not found at {css_path}")
            custom_css = "/* CSS file not found */"
    except Exception as e:
        print(f"Warning: Could not load CSS file: {e}")
        custom_css = "/* CSS file error */"
    
    with gr.Blocks(title="Beta Pose Tracker") as demo:
        # Inject custom CSS via HTML component
        # Escape any special characters in CSS for safe HTML embedding
        css_escaped = custom_css.replace("</style>", "<\\/style>")
        gr.HTML(f"<style>{css_escaped}</style>")
        
        # Header - using HTML with inline styles
        gr.HTML(
            '<div style="color: #ffffff; padding: 20px 0; margin: 0;">'
            '<h1 style="color: #ffffff !important; margin: 0; padding: 0; border: none;">BETA POSE TRACKER</h1>'
            '<h3 style="color: #ffffff !important; margin: 10px 0 0 0; padding: 0; font-style: italic; font-weight: normal; border: none;">Advanced Motion Analysis System</h3>'
            '</div>'
        )
        
        # Create tabs
        with gr.Tabs() as tabs:
            # Tab 1: Video Analysis (Trajectory Tracking)
            with gr.Tab("📹 Trajectory Tracking", id="video_analysis"):
                gr.Markdown(
                    """
                    ### **Trajectory Tracking**
                    Analyze pose trajectories from video using MediaPipe. Track body parts and visualize movement paths.
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="Upload Video",
                            elem_classes="video-container"
                        )
                        
                        track_points_input = gr.CheckboxGroup(
                            label="🎯 Track Points",
                            choices=[
                                ("Hip Mid", "hip_mid"),
                                ("Upper Body Center", "upper_body_center"),
                                ("Head", "head"),
                                ("Left Hand", "left_hand"),
                                ("Right Hand", "right_hand"),
                                ("Left Foot", "left_foot"),
                                ("Right Foot", "right_foot"),
                            ],
                            value=["hip_mid", "left_hand", "right_hand"],  # Default selections
                            info="Select body parts to track in the video"
                        )
                        
                        max_resolution_input = gr.Dropdown(
                            label="📐 Max Resolution",
                            choices=[
                                ("Original (no resize)", ""),
                                ("1920x1080 (Full HD)", "1920x1080"),
                                ("1280x720 (HD)", "1280x720"),
                                ("854x480 (SD)", "854x480"),
                                ("640x360 (Low)", "640x360"),
                            ],
                            value="1280x720",
                            info="Larger videos will be resized to reduce file size and loading time"
                        )
                        
                        analyze_btn = gr.Button(
                            "🚀 ANALYZE",
                            variant="primary",
                            scale=1
                        )
                    
                    with gr.Column():
                        video_output = gr.Video(
                            label="Output Video with Trajectories",
                            elem_classes="video-container"
                        )
                        status_output = gr.Textbox(
                            label="📊 Status",
                            interactive=False,
                            lines=3
                        )
                
                # Instructions
                with gr.Accordion("ℹ️ Instructions", open=False):
                    gr.Markdown("""
                    1. **Upload** a video file (MP4, AVI, MOV, etc.)
                    2. **Specify** which body parts to track (default: hip_mid, left_hand, right_hand)
                    3. **Select** maximum resolution to reduce file size
                    4. **Click ANALYZE** to process the video
                    5. **Download** the output video with trajectory overlays
                    """)
            
            # Tab 2: Route Segmentation
            with gr.Tab("🗺️ Route Segmentation", id="route_segmentation"):
                gr.Markdown(
                    """
                    ### **Route Segmentation**
                    Segment and analyze movement routes from pose trajectories. Identify distinct movement patterns and phases.
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        # Video source selection
                        video_source_radio = gr.Radio(
                            label="📹 Video Source",
                            choices=[
                                ("Use video from Trajectory Tracking", "use_same"),
                                ("Upload new video", "upload_new")
                            ],
                            value="use_same",
                            info="Choose to reuse the video from Trajectory Tracking tab or upload a new one"
                        )
                        
                        # Status message for video source
                        video_source_status = gr.Textbox(
                            label="",
                            value="✅ Using video from Trajectory Tracking tab",
                            interactive=False,
                            visible=True,
                            lines=1
                        )
                        
                        # Route video input (conditionally visible)
                        route_video_input = gr.Video(
                            label="📤 Upload Video for Route Analysis",
                            elem_classes="video-container",
                            visible=False  # Hidden by default when using same video
                        )
                        
                        segmentation_btn = gr.Button(
                            "🚀 SEGMENT ROUTE",
                            variant="primary",
                            scale=1
                        )
                    
                    with gr.Column():
                        route_output = gr.Video(
                            label="📥 Segmented Route Output",
                            elem_classes="video-container"
                        )
                        route_status = gr.Textbox(
                            label="📊 Status",
                            interactive=False,
                            lines=3
                        )
                
                # Placeholder for route segmentation features
                with gr.Accordion("ℹ️ About Route Segmentation", open=False):
                    gr.Markdown("""
                    **Route Segmentation** analyzes trajectory data to identify:
                    - Movement phases and transitions
                    - Route boundaries and waypoints
                    - Movement patterns and classifications
                    
                    *This feature is currently under development.*
                    """)
        
        # Event handlers
        analyze_btn.click(
            fn=process_video,
            inputs=[video_input, track_points_input, max_resolution_input],
            outputs=[video_output, status_output],
            show_progress=True
        )
        
        # Update route video input based on source choice
        def update_video_visibility(source_choice, trajectory_video):
            """Show/hide video upload based on source choice and update status."""
            if source_choice == "use_same":
                if trajectory_video is None:
                    status_msg = "⚠️ No video uploaded in Trajectory Tracking tab yet. Please upload a video there first."
                else:
                    status_msg = "✅ Using video from Trajectory Tracking tab"
                return (
                    gr.update(visible=False, value=None),  # Hide upload
                    gr.update(value=status_msg, visible=True)  # Update status
                )
            else:
                return (
                    gr.update(visible=True, value=None),  # Show upload
                    gr.update(value="Upload a new video for route segmentation", visible=True)  # Update status
                )
        
        video_source_radio.change(
            fn=update_video_visibility,
            inputs=[video_source_radio, video_input],
            outputs=[route_video_input, video_source_status]
        )
        
        # Also update when video is uploaded in trajectory tracking tab
        def update_status_on_video_upload(vid, source_choice):
            """Update status when video is uploaded in trajectory tracking tab."""
            if source_choice == "use_same":
                if vid is None:
                    return gr.update(value="⚠️ No video uploaded in Trajectory Tracking tab yet. Please upload a video there first.")
                else:
                    return gr.update(value="✅ Using video from Trajectory Tracking tab")
            return gr.update()
        
        video_input.change(
            fn=update_status_on_video_upload,
            inputs=[video_input, video_source_radio],
            outputs=[video_source_status]
        )
        
        # Function to get the correct video for route segmentation
        def get_route_video(source_choice, trajectory_video, uploaded_video):
            """Get the video to use for route segmentation."""
            if source_choice == "use_same":
                return trajectory_video
            else:
                return uploaded_video
        
        # Wrapper function for route segmentation that handles video source
        def process_route_segmentation_wrapper(source_choice, trajectory_video, uploaded_video):
            """Wrapper to handle video source selection for route segmentation."""
            video_to_use = get_route_video(source_choice, trajectory_video, uploaded_video)
            return process_route_segmentation(video_to_use)
        
        segmentation_btn.click(
            fn=process_route_segmentation_wrapper,
            inputs=[video_source_radio, video_input, route_video_input],
            outputs=[route_output, route_status],
            show_progress=True
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


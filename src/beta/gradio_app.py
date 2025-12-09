"""
Gradio frontend for Beta pose tracking.

Run with: python -m src.beta.gradio_app
or: python src/beta.gradio_app.py
"""
import sys
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np

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

# Try to import ultralytics for color segmentation
try:
    from ultralytics import YOLO
    from ultralytics.models.fastsam import FastSAM
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Color segmentation features will not be available.")
    print("Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False
    YOLO = None
    FastSAM = None


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


# ============================================================================
# Color Segmentation Functions (from notebook)
# ============================================================================

# Global models (loaded once)
_yolo_model = None
_fast_sam_model = None

def load_segmentation_models():
    """Load YOLO and FastSAM models for color segmentation."""
    global _yolo_model, _fast_sam_model
    
    if not ULTRALYTICS_AVAILABLE:
        return False, "ultralytics package not installed"
    
    try:
        # Try to find model files in multiple locations
        script_dir = Path(__file__).parent  # src/beta/
        project_root = script_dir.parent.parent  # Project root (CoBeta/)
        current_dir = Path.cwd()  # Current working directory (repo root on Hugging Face Spaces)
        
        # Check if we're running on Hugging Face Spaces
        is_hf_space = os.getenv("SPACE_ID") is not None or os.getenv("SYSTEM") == "spaces"
        
        # List of possible locations to search
        # For Hugging Face Spaces, prioritize repo root and models/ directory
        if is_hf_space:
            search_paths = [
                current_dir,  # Repo root (most common for HF Spaces)
                current_dir / "models",  # models/ directory
                current_dir / "src" / "beta",  # src/beta/ 
                script_dir,  # src/beta/ (fallback)
            ]
        else:
            # Local development paths
            search_paths = [
                script_dir,  # src/beta/best_color.pt
                script_dir.parent,  # src/best_color.pt
                project_root,  # CoBeta/best_color.pt
                current_dir,  # Current working directory
                Path.home() / ".cache" / "cobeta",  # Cache directory
            ]
        
        # Allow override via environment variable
        if os.getenv("BEST_COLOR_MODEL_PATH"):
            search_paths.insert(0, Path(os.getenv("BEST_COLOR_MODEL_PATH")).parent)
        
        yolo_model_path = None
        fastsam_model_path = None
        
        # Search for YOLO model
        for search_path in search_paths:
            potential_path = search_path / "best_color.pt"
            if potential_path.exists():
                yolo_model_path = potential_path
                break
        
        # Search for FastSAM model
        for search_path in search_paths:
            potential_path = search_path / "FastSAM-x.pt"
            if potential_path.exists():
                fastsam_model_path = potential_path
                break
        
        # If not found, try alternative FastSAM names
        if fastsam_model_path is None:
            for search_path in search_paths:
                for alt_name in ["FastSAM.pt", "fastsam.pt", "FastSAM-s.pt"]:
                    potential_path = search_path / alt_name
                    if potential_path.exists():
                        fastsam_model_path = potential_path
                        break
                if fastsam_model_path:
                    break
        
        # Build error message with all searched locations
        searched_locations = "\n".join([f"  - {p / 'best_color.pt'}" for p in search_paths])
        
        if yolo_model_path is None:
            return False, (
                f"YOLO model (best_color.pt) not found.\n\n"
                f"Searched in:\n{searched_locations}\n\n"
                f"Please download best_color.pt from Google Drive:\n"
                f"https://drive.google.com/uc?id=1fjnYNgxwBC_vi9XNU9szcXzhq-nzKq5e\n\n"
                f"Place it in one of the above locations (recommended: {script_dir} or {project_root})"
            )
        
        # If FastSAM not found, try to auto-download it
        if fastsam_model_path is None:
            print("FastSAM model not found locally. Attempting to download...")
            try:
                # FastSAM can auto-download if we just specify the model name
                fastsam_model_path = "FastSAM-x.pt"  # This will trigger auto-download
                print("Downloading FastSAM-x.pt (this may take a few minutes)...")
            except Exception as e:
                return False, (
                    f"FastSAM model (FastSAM-x.pt) not found and auto-download failed.\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Please download FastSAM-x.pt manually or ensure you have internet connection."
                )
        
        print(f"Loading YOLO model from: {yolo_model_path}")
        _yolo_model = YOLO(str(yolo_model_path))
        
        print(f"Loading FastSAM model from: {fastsam_model_path}")
        # FastSAM will auto-download if the file doesn't exist locally
        _fast_sam_model = FastSAM(str(fastsam_model_path))
        
        print("Models loaded successfully!")
        return True, f"Models loaded successfully!\nYOLO: {yolo_model_path}\nFastSAM: {fastsam_model_path}"
        
    except Exception as e:
        return False, f"Error loading models: {str(e)}"


def calculate_iou(box, mask_box):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1 = max(box[0], mask_box[0])
    y1 = max(box[1], mask_box[1])
    x2 = min(box[2], mask_box[2])
    y2 = min(box[3], mask_box[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box[2]-box[0])*(box[3]-box[1])) + ((mask_box[2]-mask_box[0])*(mask_box[3]-mask_box[1])) - intersection
    return intersection / union if union > 0 else 0


def run_combined_inference(image, target_colors=None):
    """
    Run combined YOLO + FastSAM inference for color segmentation.
    
    Args:
        image: Input image (BGR format)
        target_colors: List of color names to filter, or None for all colors
    
    Returns:
        Annotated image with detections and masks
    """
    if _yolo_model is None or _fast_sam_model is None:
        return image
    
    # YOLO detection
    yolo_results = _yolo_model(image, verbose=False, conf=0.5)[0]
    if len(yolo_results.boxes) == 0:
        return image
    
    yolo_boxes = yolo_results.boxes.xyxy.cpu().numpy()
    class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
    class_names = yolo_results.names
    
    # FastSAM segmentation
    fast_results = _fast_sam_model(image, retina_masks=True, imgsz=640, conf=0.3, iou=0.6, verbose=False)[0]
    
    annotated_frame = image.copy()
    
    # Convert target_colors to set for faster lookup
    if target_colors:
        target_colors = set(target_colors)
    
    # If FastSAM didn't find any masks, just draw YOLO boxes
    if fast_results.masks is None:
        for i, box in enumerate(yolo_boxes):
            label = class_names[class_ids[i]]
            # Check if label is in target colors (or target is None for all)
            if (target_colors is None) or (label in target_colors):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return annotated_frame
    
    # YOLO + FastSAM matching
    all_masks = fast_results.masks.data.cpu().numpy()
    all_mask_boxes = fast_results.boxes.xyxy.cpu().numpy()
    
    for i, target_box in enumerate(yolo_boxes):
        label = class_names[class_ids[i]]
        
        # Filter color logic: if target_colors has value and label not in list, skip
        if target_colors and label not in target_colors:
            continue
        
        # Find best matching mask by IoU
        best_iou, best_mask_idx = 0, -1
        for j, mask_box in enumerate(all_mask_boxes):
            iou = calculate_iou(target_box, mask_box)
            if iou > best_iou:
                best_iou = iou
                best_mask_idx = j
        
        if best_iou > 0.5 and best_mask_idx != -1:
            # Draw mask overlay
            mask = all_masks[best_mask_idx]
            if mask.shape[:2] != annotated_frame.shape[:2]:
                mask = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
            
            color = [0, 255, 0]  # Green mask
            annotated_frame[mask > 0.5] = annotated_frame[mask > 0.5] * 0.5 + np.array(color) * 0.5
            
            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, target_box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No matching mask, just draw box
            x1, y1, x2, y2 = map(int, target_box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame


def get_target_colors(selected_colors):
    """
    Helper function to determine target colors from user selection.
    
    Args:
        selected_colors: List of selected colors from UI
    
    Returns:
        None if "All" selected or empty, otherwise list of selected colors
    """
    if not selected_colors:
        return None
    if "All" in selected_colors:
        return None
    return selected_colors


def process_segmentation_image(image, selected_colors):
    """Process image for color segmentation."""
    if image is None:
        return None
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    target = get_target_colors(selected_colors)
    result_bgr = run_combined_inference(image_bgr, target_colors=target)
    # Convert back to RGB for Gradio
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def process_segmentation_video(video_path, selected_colors):
    """Process video for color segmentation."""
    if video_path is None:
        return None
    
    # Get video path
    if isinstance(video_path, str):
        video_file = video_path
    elif hasattr(video_path, 'name'):
        video_file = video_path.name
    else:
        video_file = str(video_path)
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out_path = temp_output.name
    temp_output.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    target = get_target_colors(selected_colors)
    
    frame_count = 0
    MAX_FRAMES = 300  # Limit processing to prevent timeout
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count >= MAX_FRAMES:
                break
            
            result_frame = run_combined_inference(frame, target_colors=target)
            out.write(result_frame)
            frame_count += 1
    finally:
        cap.release()
        out.release()
    
    return out_path


# Placeholder function for route segmentation (kept for backward compatibility)
def process_route_segmentation(video_file):
    """Process route segmentation with color detection."""
    if not ULTRALYTICS_AVAILABLE:
        return None, "Error: ultralytics package not installed. Install with: pip install ultralytics"
    
    # Ensure models are loaded
    if _yolo_model is None or _fast_sam_model is None:
        success, msg = load_segmentation_models()
        if not success:
            return None, f"Error loading models: {msg}"
    
    if video_file is None:
        return None, "Please upload a video file for route segmentation."
    
    try:
        output_path = process_segmentation_video(video_file, None)  # Process all colors by default
        if output_path:
            return output_path, f"Successfully processed video with color segmentation!"
        else:
            return None, "Error processing video."
    except Exception as e:
        return None, f"Error during processing: {str(e)}"


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
            
            # Tab 2: Route Segmentation (Color Detection & Segmentation)
            with gr.Tab("🗺️ Route Segmentation", id="route_segmentation"):
                gr.Markdown(
                    """
                    ### **Route Segmentation - Color Detection & Segmentation**
                    Detect and segment climbing holds by color using YOLO and FastSAM. Upload an image or video to analyze.
                    """
                )
                
                # Define color choices (same as notebook)
                specific_colors = ['black', 'blue', 'brown', 'cream', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
                ui_choices = ["All"] + specific_colors
                
                # Load models on startup
                with gr.Row():
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="Loading models...",
                        interactive=False,
                        lines=2
                    )
                
                # Create tabs for Image and Video modes
                with gr.Tabs() as seg_tabs:
                    # Image Mode
                    with gr.TabItem("Image Mode"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                seg_img_input = gr.Image(
                                    label="Upload Image",
                                    type="numpy"
                                )
                                seg_img_color = gr.CheckboxGroup(
                                    choices=ui_choices,
                                    label="Target Colors",
                                    info="Select 'All' to detect all colors in the list.",
                                    value=["All"]
                                )
                                seg_img_btn = gr.Button(
                                    "Analyze Image",
                                    variant="primary"
                                )
                            with gr.Column(scale=2):
                                seg_img_output = gr.Image(
                                    label="Result"
                                )
                    
                    # Video Mode
                    with gr.TabItem("Video Mode"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Video source selection
                                seg_video_source_radio = gr.Radio(
                                    label="📹 Video Source",
                                    choices=[
                                        ("Use video from Trajectory Tracking", "use_same"),
                                        ("Upload new video", "upload_new")
                                    ],
                                    value="use_same",
                                    info="Choose to reuse the video from Trajectory Tracking tab or upload a new one"
                                )
                                
                                # Status message for video source
                                seg_video_source_status = gr.Textbox(
                                    label="",
                                    value="✅ Using video from Trajectory Tracking tab",
                                    interactive=False,
                                    visible=True,
                                    lines=1
                                )
                                
                                # Route video input (conditionally visible)
                                seg_video_input = gr.Video(
                                    label="📤 Upload Video for Route Analysis",
                                    elem_classes="video-container",
                                    visible=False  # Hidden by default when using same video
                                )
                                
                                seg_vid_color = gr.CheckboxGroup(
                                    choices=ui_choices,
                                    label="Target Colors",
                                    info="Select 'All' to detect all colors in the list.",
                                    value=["All"]
                                )
                                
                                seg_vid_btn = gr.Button(
                                    "Analyze Video",
                                    variant="primary"
                                )
                            with gr.Column(scale=2):
                                seg_vid_output = gr.Video(
                                    label="Result"
                                )
                                seg_vid_status = gr.Textbox(
                                    label="📊 Status",
                                    interactive=False,
                                    lines=3
                                )
                
                # Instructions
                with gr.Accordion("ℹ️ About Route Segmentation", open=False):
                    gr.Markdown("""
                    **Route Segmentation** uses YOLO and FastSAM to detect and segment climbing holds by color:
                    - **YOLO** detects colored holds with bounding boxes
                    - **FastSAM** provides precise segmentation masks
                    - **Color Filtering** allows you to focus on specific hold colors
                    
                    **Available Colors:** black, blue, brown, cream, green, orange, pink, purple, red, white, yellow
                    
                    **Note:** Models (best_color.pt and FastSAM-x.pt) must be in the same directory as this script.
                    """)
        
        # Event handlers
        analyze_btn.click(
            fn=process_video,
            inputs=[video_input, track_points_input, max_resolution_input],
            outputs=[video_output, status_output],
            show_progress=True
        )
        
        # Load models on startup for segmentation
        def load_models_on_startup():
            """Load segmentation models and return status message."""
            success, msg = load_segmentation_models()
            if success:
                return f"✅ {msg}"
            else:
                return f"⚠️ {msg}\n\nPlease ensure best_color.pt and FastSAM-x.pt are available."
        
        # Update model status on load
        demo.load(load_models_on_startup, outputs=[model_status])
        
        # Image segmentation event handler
        seg_img_btn.click(
            fn=process_segmentation_image,
            inputs=[seg_img_input, seg_img_color],
            outputs=[seg_img_output],
            show_progress=True
        )
        
        # Video segmentation - update video visibility based on source choice
        def update_seg_video_visibility(source_choice, trajectory_video):
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
        
        seg_video_source_radio.change(
            fn=update_seg_video_visibility,
            inputs=[seg_video_source_radio, video_input],
            outputs=[seg_video_input, seg_video_source_status]
        )
        
        # Update status when video is uploaded in trajectory tracking tab
        def update_seg_status_on_video_upload(vid, source_choice):
            """Update status when video is uploaded in trajectory tracking tab."""
            if source_choice == "use_same":
                if vid is None:
                    return gr.update(value="⚠️ No video uploaded in Trajectory Tracking tab yet. Please upload a video there first.")
                else:
                    return gr.update(value="✅ Using video from Trajectory Tracking tab")
            return gr.update()
        
        video_input.change(
            fn=update_seg_status_on_video_upload,
            inputs=[video_input, seg_video_source_radio],
            outputs=[seg_video_source_status]
        )
        
        # Function to get the correct video for segmentation
        def get_seg_video(source_choice, trajectory_video, uploaded_video):
            """Get the video to use for segmentation."""
            if source_choice == "use_same":
                return trajectory_video
            else:
                return uploaded_video
        
        # Wrapper function for video segmentation that handles video source and colors
        def process_segmentation_video_wrapper(source_choice, trajectory_video, uploaded_video, selected_colors):
            """Wrapper to handle video source selection and color filtering for segmentation."""
            video_to_use = get_seg_video(source_choice, trajectory_video, uploaded_video)
            if video_to_use is None:
                return None, "Please upload a video file or select a video from Trajectory Tracking tab."
            
            if not ULTRALYTICS_AVAILABLE:
                return None, "Error: ultralytics package not installed. Install with: pip install ultralytics"
            
            # Ensure models are loaded
            if _yolo_model is None or _fast_sam_model is None:
                success, msg = load_segmentation_models()
                if not success:
                    return None, f"Error loading models: {msg}"
            
            try:
                output_path = process_segmentation_video(video_to_use, selected_colors)
                if output_path:
                    return output_path, f"Successfully processed video with color segmentation! Colors: {', '.join(selected_colors) if selected_colors else 'All'}"
                else:
                    return None, "Error processing video."
            except Exception as e:
                return None, f"Error during processing: {str(e)}"
        
        seg_vid_btn.click(
            fn=process_segmentation_video_wrapper,
            inputs=[seg_video_source_radio, video_input, seg_video_input, seg_vid_color],
            outputs=[seg_vid_output, seg_vid_status],
            show_progress=True
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


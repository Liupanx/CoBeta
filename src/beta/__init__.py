import cv2
from pathlib import Path
from typing import Optional, Callable, Union, List, Dict, Tuple
import mediapipe as mp
from utils.pose_helper import get_track_point_coords
from utils.draw_helper import draw_trajectory, draw_velocity_arrow
from utils.smooth_filter import smooth_point


class Beta:
    """Video helper with MediaPipe pose analysis built in.

    Usage:
      beta = Beta()
      beta.load_video(path, show=True)

    The method can optionally accept an `on_landmarks` callback which will
    receive the `results.pose_landmarks` object for each successfully-read
    frame. This is useful for headless processing.
    """

    def __init__(self, track_points: Optional[List[str]] = None) -> None:
        """Initialize Beta tracker.
        
        Args:
            track_points: List of body parts to track (e.g., ["hip_mid", "left_hand", "right_hand"])
                          If None, defaults to tracking hip center only.
        """
        print("Beta class initialized")
        
        # Default to tracking hip center if nothing specified
        if track_points is None:
            track_points = ["hip_mid", "left_hand", "right_hand"]
        
        # Store which points we want to track
        self.track_points = track_points

        # for testing smoothing
        # Store last smoothed point for each body part
        self.smoothed_points: Dict[str, Optional[Tuple[int, int]]] = {
            point: None for point in track_points
        }

        
        # Initialize trajectory storage: dictionary where each key is a body part
        # and value is a list of (x, y) coordinates over time
        # Example: {"hip_mid": [(100, 200), (105, 205), ...], "left_hand": [...]}
        self.trajectories: Dict[str, List[Tuple[int, int]]] = {
            point: [] for point in track_points
        }
        
        # Color mapping for different track points (BGR format for OpenCV)
        self.track_point_colors: Dict[str, Tuple[int, int, int]] = {
            "hip_mid": (0, 255, 255),        # Yellow
            "upper_body_center": (255, 255, 0),  # Cyan
            "head": (255, 0, 255),           # Magenta
            "left_hand": (255, 0, 0),        # Blue
            "right_hand": (0, 0, 255),       # Red
            "left_foot": (0, 255, 0),        # Green
            "right_foot": (0, 128, 255),     # Orange
        }



    def load_video(
        self,
        file_path: Union[str, Path],
        *,
        show: bool = True,
        on_landmarks: Optional[Callable[..., None]] = None,
    ) -> bool:
        """Open video, run MediaPipe Pose per frame, draw landmarks, and show.

        Args:
            file_path: path to video file (str or Path).
            show: if True, display frames with cv2.imshow. Set False for
                headless environments.
            on_landmarks: optional callback called as `on_landmarks(landmarks, frame)`
                where `landmarks` may be None if no pose was found.

        Returns:
            True on success (opened and processed), False on failure.
        """

        path = Path(file_path)
        print(f"Trying to open: {path.resolve()}")
        if not path.exists():
            print(f"File not found: {path.resolve()}")
            return False

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print("Error opening video stream or file")
            return False

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # Use context manager for correct resource handling inside MediaPipe
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    # TODO: check the video ratio and determin rotate or not.
                    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    if not ret:
                        # end of stream or read error
                        break

                    # MediaPipe expects RGB images
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    # Draw landmarks on the original BGR frame for display
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                        )
                        
                        # Extract and store trajectory points
                        # Get frame dimensions for coordinate conversion
                        frame_height, frame_width = frame.shape[:2]
                        
                        # Extract and track each body part using pose_helper
                        for track_point in self.track_points:
                            result = get_track_point_coords(
                                track_point,
                                results.pose_landmarks.landmark,
                                frame_width,
                                frame_height
                            )
                            if result is not None:
                                mid_point, _mid_point_3d = result

                                # 1) get previous smoothed point for this track_point
                                prev_smoothed = self.smoothed_points.get(track_point)

                                # 2) compute new smoothed point
                                # alpha parameter, the smaller the smoother but slower.
                                smoothed_point = smooth_point(prev_smoothed, mid_point, 0.3)

                                # 3) store it back so next frame can use it
                                self.smoothed_points[track_point] = smoothed_point

                                # 4) use the smoothed point for trajectory
                                self.trajectories[track_point].append(smoothed_point)

                                color = self.track_point_colors.get(track_point, (255, 255, 255))

                                # Draw the smoothed point
                                cv2.circle(frame, smoothed_point, 8, color, -1)

                                # Draw trajectory (optionally only recent history)
                                trajectory = self.trajectories[track_point]
                                if len(trajectory) > 1:
                                    draw_trajectory(frame, trajectory, color, thickness=2)
                            
                            # if result is not None:
                            #     mid_point, _mid_point_3d = result
                                
                            #     # Add this point to our trajectory
                            #     self.trajectories[track_point].append(mid_point)
                                
                            #     # Get color for this track point (default to white if not found)
                            #     color = self.track_point_colors.get(track_point, (255, 255, 255))
                                
                            #     # Draw the track point
                            #     cv2.circle(frame, mid_point, 8, color, -1)  # Filled circle
                                
                            #     # Draw the trajectory path using draw_helper
                            #     trajectory = self.trajectories[track_point]
                            #     if len(trajectory) > 1:
                            #         draw_trajectory(frame, trajectory, color, thickness=2)
                                    
                            #         # (optional) draw velocity arrow (shows direction of movement)
                            #         # draw_velocity_arrow(frame, trajectory[-2], trajectory[-1], color, scale=3, thickness=3)
                                

                    # callback for downstream processing (e.g., logging or ML)
                    if on_landmarks:
                        try:
                            on_landmarks(results.pose_landmarks, frame)
                        except Exception as e:
                            print("on_landmarks callback error:", e)

                    if show:
                        try:
                            cv2.imshow("Frame", frame)
                            if cv2.waitKey(25) & 0xFF == ord("q"):
                                break
                        except cv2.error:
                            # likely headless; stop showing but continue
                            print(
                                "cv2.imshow failed (headless?),"
                                " continuing without display"
                            )
                            show = False
            finally:
                cap.release()
                if show:
                    cv2.destroyAllWindows()

        return True


if __name__ == "__main__":
    # example usage: adjust the path below or run this module directly
    # Default tracks: hip_mid, left_hand, right_hand
    # Or specify custom: Beta(track_points=["hip_mid"])
    beta = Beta()  # Uses default: ["hip_mid", "left_hand", "right_hand"]
    beta.load_video(
        "/Users/claireliu/Desktop/Project_25/CoBeta/src/beta/example.mp4",
        show=True,
    )
    # After processing, you can access trajectories:
    print(f"Hip trajectory has {len(beta.trajectories['hip_mid'])} points")
    print(f"Left hand trajectory has {len(beta.trajectories['left_hand'])} points")
    print(f"Right hand trajectory has {len(beta.trajectories['right_hand'])} points")


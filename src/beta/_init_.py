import cv2
from pathlib import Path
from typing import Optional, Callable, Union, List, Dict, Tuple
import mediapipe as mp


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
        
        # Initialize trajectory storage: dictionary where each key is a body part
        # and value is a list of (x, y) coordinates over time
        # Example: {"hip_mid": [(100, 200), (105, 205), ...], "left_hand": [...]}
        self.trajectories: Dict[str, List[Tuple[int, int]]] = {
            point: [] for point in track_points
        }

    def _get_hip_center(self, landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """Extract the center point between left and right hips.
        
        MediaPipe landmarks are in normalized coordinates (0-1), so we need to
        multiply by frame dimensions to get pixel coordinates.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
            
        Returns:
            (x, y) tuple of hip center in pixel coordinates, or None if not found
        """
        mp_pose = mp.solutions.pose
        
        # Get left and right hip landmarks
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Check visibility (MediaPipe provides a visibility score 0-1)
        # We only use landmarks that are reasonably visible
        if left_hip.visibility < 0.5 or right_hip.visibility < 0.5:
            return None

        # Calculate center point
        # MediaPipe coordinates are normalized (0-1), so multiply by frame size
        center_x = int((left_hip.x + right_hip.x) / 2 * frame_width)
        center_y = int((left_hip.y + right_hip.y) / 2 * frame_height)

        
        return (center_x, center_y)

    def _get_left_hand(self, landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """Extract the left hand landmark.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
        """
        mp_pose = mp.solutions.pose
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        if left_hand.visibility < 0.5:
            return None
        return (int(left_hand.x * frame_width), int(left_hand.y * frame_height))

    def _get_right_hand(self, landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """Extract the right hand landmark.
        """
        mp_pose = mp.solutions.pose
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        if right_hand.visibility < 0.5:
            return None
        return (int(right_hand.x * frame_width), int(right_hand.y * frame_height))

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
                        
                        # Extract and track each body part separately
                        # Hip center
                        if "hip_mid" in self.track_points:
                            hip_center = self._get_hip_center(
                                results.pose_landmarks.landmark,
                                frame_width,
                                frame_height
                            )
                            if hip_center is not None:
                                # Add this point to our trajectory
                                self.trajectories["hip_mid"].append(hip_center)
                                # Draw the hip_mid point
                                cv2.circle(frame, hip_center, 8, (0, 255, 255), -1)  # Yellow circle
                                
                                # Draw the trajectory path
                                trajectory = self.trajectories["hip_mid"]
                                if len(trajectory) > 1:
                                    for i in range(1, len(trajectory)):
                                        cv2.line(
                                            frame,
                                            trajectory[i-1],
                                            trajectory[i],
                                            (0, 255, 255),  # Yellow line
                                            2  # Line thickness
                                        )
                        
                        # Left hand
                        if "left_hand" in self.track_points:
                            left_hand = self._get_left_hand(
                                results.pose_landmarks.landmark,
                                frame_width,
                                frame_height
                            )
                            if left_hand is not None:  # Only append if not None!
                                self.trajectories["left_hand"].append(left_hand)
                                # Draw the left hand point
                                cv2.circle(frame, left_hand, 8, (255, 0, 0), -1)  # Blue circle
                                
                                # Draw the trajectory path
                                left_hand_trajectory = self.trajectories["left_hand"]
                                if len(left_hand_trajectory) > 1:
                                    for i in range(1, len(left_hand_trajectory)):
                                        cv2.line(
                                            frame,
                                            left_hand_trajectory[i-1],
                                            left_hand_trajectory[i],
                                            (255, 0, 0),  # Blue line
                                            2  # Line thickness
                                        )
                        
                        # Right hand
                        if "right_hand" in self.track_points:
                            right_hand = self._get_right_hand(
                                results.pose_landmarks.landmark,
                                frame_width,
                                frame_height
                            )
                            if right_hand is not None:  # Only append if not None!
                                self.trajectories["right_hand"].append(right_hand)
                                # Draw the right hand point
                                cv2.circle(frame, right_hand, 8, (0, 0, 255), -1)  # Red circle
                                
                                # Draw the trajectory path
                                right_hand_trajectory = self.trajectories["right_hand"]
                                if len(right_hand_trajectory) > 1:
                                    for i in range(1, len(right_hand_trajectory)):
                                        cv2.line(
                                            frame,
                                            right_hand_trajectory[i-1],
                                            right_hand_trajectory[i],
                                            (0, 0, 255),  # Red line
                                            2  # Line thickness
                                        )
                                

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


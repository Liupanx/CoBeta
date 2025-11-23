import mediapipe as mp

def get_track_point_coords(tp, landmark):
    """Convert MediaPipe pose landmarks to a dictionary of body part coordinates.
    """
    mp_pose = mp.solutions.pose
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    if tp == "hip_mid":
        if (
            left_hip.visibility < confidence_threshold
            or right_hip.visibility < confidence_threshold
        ):
            return None
import mediapipe as mp

def get_track_point_coords(tp, landmarks, frame_width, frame_height, confidence_threshold=0.5):
    """Convert MediaPipe pose landmarks to a dictionary of body part coordinates.
    Returns:
        mid_point: (x, y) tuple of the mid point in pixel coordinates (for drawing display)
        mid_point_3d: (x, y, z) tuple of the mid point in 3D coordinates
        None if the mid point is not found or the confidence threshold is not met
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
        if (left_hip.visibility < confidence_threshold or right_hip.visibility < confidence_threshold):
            return None
        left_hip = (int(left_hip.x * frame_width), int(left_hip.y * frame_height))
        right_hip = (int(right_hip.x * frame_width), int(right_hip.y * frame_height))

        mid_point = (int((left_hip[0] + right_hip[0]) / 2), int((left_hip[1] + right_hip[1]) / 2))
        mid_point_3d = (
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2
        )
    if tp == "upper_body_center":
        if (
            left_hip.visibility < confidence_threshold
            or right_hip.visibility < confidence_threshold
            or left_shoulder.visibility < confidence_threshold
            or right_shoulder.visibility < confidence_threshold
        ):
            return None
        pts_2d = [
            (int(left_hip.x * frame_width), int(left_hip.y * frame_height)),
            (int(right_hip.x * frame_width), int(right_hip.y * frame_height)),
            (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height)),
            (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height)),
        ]
        mid_point = (
            int(sum(p[0] for p in pts_2d) / 4),
            int(sum(p[1] for p in pts_2d) / 4),
        )
        mid_point_3d = (
            (left_hip.x + right_hip.x + left_shoulder.x + right_shoulder.x) / 4,
            (left_hip.y + right_hip.y + left_shoulder.y + right_shoulder.y) / 4,
            (left_hip.z + right_hip.z + left_shoulder.z + right_shoulder.z) / 4,
        )
        if tp == "head":
            if (nose.visibility < confidence_threshold):
                return None
        mid_point = (int(nose.x * frame_width), int(nose.y * frame_height))
        mid_point_3d = (nose.x, nose.y, nose.z)

        # TODO: Add more track points here
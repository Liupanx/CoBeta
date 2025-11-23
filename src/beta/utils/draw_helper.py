import cv2

def draw_trajectory(frame, trajectories, color, thickness=2):
    "Function use to draw connected points into trajectory lines."
    for i in range(1, len(trajectories)):
        cv2.line(frame, trajectories[i-1], trajectories[i], color, thickness)


def draw_velocity_arrow(frame, prev_point, curr_point, color, scale=5, thickness=3):
    x1, y1 = prev_point
    x2, y2 = curr_point

    dx = x2 - x1
    dy = y2 - y1

    end_x = x2 + (dx * scale)
    end_y = y2 + (dy * scale)
    end_point = (end_x, end_y)

    cv2.arrowedLine(frame, curr_point, end_point, color, thickness, tipLength=0.3)

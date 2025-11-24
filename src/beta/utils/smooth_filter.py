from typing import Optional, Tuple

def smooth_point(
        prev_smoothed: Optional[Tuple[int, int]],
        new_point: Tuple[int, int],
        alpha: float = 0.3,
        ) -> Tuple[int, int]:

        if prev_smoothed is None:
            # first frame: nothing to smooth yet
            return new_point

        x_prev, y_prev = prev_smoothed
        x_new, y_new = new_point

        x_s = alpha * x_new + (1 - alpha) * x_prev
        y_s = alpha * y_new + (1 - alpha) * y_prev

        return int(x_s), int(y_s)
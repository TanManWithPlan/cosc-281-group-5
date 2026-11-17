"""Road-relative proportional control. Positive offset is road right.

Outputs are forward m/s and left-positive yaw rad/s. No ROS dependency.
Stop immediately when detection is missing, even if estimation holds its value.
"""
import math


class RoadController:
    def __init__(self, forward_speed=0.05, kp=0.8, max_yaw_rate=0.6,
                 turn_slowdown=0.8, max_image_age_s=0.25):
        values = (forward_speed, kp, max_yaw_rate, turn_slowdown, max_image_age_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('controller parameters must be finite')
        if (forward_speed < 0 or kp < 0 or max_yaw_rate <= 0 or
                not 0 <= turn_slowdown <= 1 or max_image_age_s <= 0):
            raise ValueError('invalid controller parameters')
        self.forward_speed = forward_speed
        self.kp = kp
        self.max_yaw_rate = max_yaw_rate
        self.turn_slowdown = turn_slowdown
        self.max_image_age_s = max_image_age_s

    def compute(self, estimate, road_found=True, boundary_warning=False):
        if estimate is None or not estimate.valid or not road_found or boundary_warning:
            return 0., 0.
        offset, age = estimate.offset, estimate.image_age_s
        if (offset is None or not math.isfinite(offset) or abs(offset) > 1 or
                not math.isfinite(age) or not 0 <= age <= self.max_image_age_s):
            return 0., 0.
        # Road right -> turn right -> negative ROS yaw.
        yaw = max(-self.max_yaw_rate, min(self.max_yaw_rate, -self.kp * offset))
        forward = self.forward_speed * (1 - self.turn_slowdown * abs(offset))
        return forward, yaw

"""Navigation node: camera -> perception -> estimation -> control -> request.

Wiring of the team's modules into the four starter hooks:

    process_image    perception.Perception     one frame -> PerceptionResult
    update_estimate  estimation.RoadEstimator  noisy per-frame offset -> smoothed, validated
    choose_action    (this file)               'follow' when the estimate is valid, else stop
    compute_motion   (this file)               PLACEHOLDER proportional controller (Control owner replaces)

Sign conventions (they differ, so mind the minus sign in compute_motion):
    perception / estimate offset:  -1 .. +1, POSITIVE = road is to the RIGHT of centre
    MotionRequest yaw rate:        POSITIVE = turn LEFT (rad/s)
"""
import math
from typing import NamedTuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from course_robot_interfaces.msg import MotionRequest

from team_navigation.perception import Perception, load_config
from team_navigation.estimation import RoadEstimator


class Action(NamedTuple):
    mode: str       # 'follow' for now; room for 'take_fork', 'recover', ...
    offset: float   # smoothed road offset, -1..+1, + = road to the right


class Navigation(Node):
    def __init__(self):
        super().__init__('navigation')
        p = self.declare_parameter
        self.target_speed = p('requested_forward_speed_m_s', .05).value
        self.yaw_gain = p('yaw_gain_rad_s', 1.0).value      # rad/s per unit of offset
        self.max_yaw = p('max_yaw_rad_s', 1.0).value
        # Same file the track_test web page writes with "Save to file".
        hsv_path = p('hsv_config_path', '~/hsv_config.json').value
        self.perception = Perception(load_config(hsv_path))
        self.estimator = RoadEstimator(
            smoothing_alpha=p('smoothing_alpha', .2).value,
            max_image_age_s=p('max_image_age_s', .25).value,
            road_lost_timeout_s=p('road_lost_timeout_s', .5).value,
            max_jump=p('max_jump', .3).value,
            outlier_accept_count=p('outlier_accept_count', 3).value)
        self.bridge = CvBridge()
        self.image = self.image_stamp = self.imu = None
        self.previous_error = None
        self._processed_key = self._observation = None
        self._was_valid = None
        self.create_subscription(Image, 'camera/image_raw', self.image_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, 'imu/data', self.imu_cb, qos_profile_sensor_data)
        self.publisher = self.create_publisher(
            MotionRequest, 'navigation/request',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(.05, self.tick)

    def image_cb(self, msg):
        try:
            # OpenCV uses BGR channel order. The image is an H x W x 3 array.
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_stamp = msg.header.stamp
        except Exception as exc:
            self.image = self.image_stamp = None
            self.get_logger().error(f'Image conversion failed: {exc}')

    def imu_cb(self, msg):
        self.imu = msg

    def process_image(self, image):
        """Return a perception.PerceptionResult for the newest frame, or None without an image.

        The 20 Hz timer is not synchronised with the camera, so the same frame
        is often seen twice; reuse the previous result instead of reprocessing.
        """
        if image is None:
            return None
        stamp = self.image_stamp
        key = (stamp.sec, stamp.nanosec) if stamp is not None else None
        if key is None or key != self._processed_key:
            self._observation = self.perception.process(image)
            self._processed_key = key
        return self._observation

    def update_estimate(self, observation, imu):
        """Return an estimation.RoadEstimate (smoothed offset + valid flag)."""
        now_s = self.get_clock().now().nanoseconds / 1e9
        if observation is None or self.image_stamp is None:
            estimate = self.estimator.update(0., False, None, now_s)
        else:
            stamp_s = self.image_stamp.sec + self.image_stamp.nanosec * 1e-9
            estimate = self.estimator.update(observation.error, observation.found, stamp_s, now_s)
        if estimate.valid != self._was_valid:   # log transitions only, not every tick
            self._was_valid = estimate.valid
            self.get_logger().info(
                f'road estimate {"valid" if estimate.valid else "INVALID"}: {estimate.status}')
        return estimate

    def choose_action(self, estimate):
        """Follow the road while the estimate is usable; otherwise no action (stop)."""
        if estimate is None or not estimate.valid:
            return None
        return Action('follow', estimate.offset)

    def compute_motion(self, action):
        """PLACEHOLDER proportional controller; the Control owner replaces this.

        Returns (forward m/s, left-turn rad/s). Slows a little when far off-centre.
        """
        if action is None:
            return 0., 0.
        yaw = -self.yaw_gain * action.offset          # road right (+) -> turn right (-)
        yaw = max(-self.max_yaw, min(self.max_yaw, yaw))
        forward = self.target_speed * (1. - .5 * min(abs(action.offset), 1.))
        return forward, yaw

    def tick(self):
        forward, yaw = 0., 0.
        try:
            observation = self.process_image(self.image)
            estimate = self.update_estimate(observation, self.imu)
            action = self.choose_action(estimate)
            forward, yaw = self.compute_motion(action)
            if not all(math.isfinite(v) for v in (forward, yaw)):
                raise ValueError('controller returned a nonfinite command')
            self.previous_error = None
        except Exception as exc:
            forward, yaw = 0., 0.
            error = str(exc)
            if error != self.previous_error:
                self.get_logger().error(f'Navigation failed; requesting zero motion: {error}')
                self.previous_error = error
        msg = MotionRequest()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.forward_speed_m_s, msg.yaw_rate_rad_s = float(forward), float(yaw)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = Navigation()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

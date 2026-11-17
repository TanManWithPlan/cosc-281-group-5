"""Receive sensors and request motion through the supplied safety node.

Integrates team perception and estimation with bounded proportional control.
"""
import math
from team_navigation.control import RoadController
from team_navigation.estimation import RoadEstimator
from team_navigation.perception import Perception, DEFAULT_CONFIG
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from course_robot_interfaces.msg import MotionRequest


class Navigation(Node):
    def __init__(self):
        super().__init__('navigation')
        self.target_speed = self.declare_parameter('requested_forward_speed_m_s', .05).value
        self.controller = RoadController(
            forward_speed=self.target_speed,
            kp=self.declare_parameter('steering_kp', 0.8).value,
            max_yaw_rate=self.declare_parameter('max_yaw_rate_rad_s', 0.6).value,
            turn_slowdown=self.declare_parameter('turn_slowdown', 0.8).value,
            max_image_age_s=self.declare_parameter('max_image_age_s', 0.25).value)
        self.estimator = RoadEstimator(max_image_age_s=self.controller.max_image_age_s)
        config = {key: list(self.declare_parameter(key, value).value)
                  for key, value in DEFAULT_CONFIG.items()}
        self.perception = Perception(config)
        self.observation = None
        self.processed_stamp = None
        self.bridge = CvBridge()
        self.image = self.image_stamp = self.imu = None
        self.previous_error = None
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
        """Process each captured frame once; never reuse a failed conversion."""
        if image is None or self.image_stamp is None:
            self.observation = None
            return None
        stamp = (self.image_stamp.sec, self.image_stamp.nanosec)
        if stamp != self.processed_stamp:
            self.observation = self.perception.process(image)
            self.processed_stamp = stamp
        return self.observation

    def update_estimate(self, observation, imu):
        """Smooth road-relative image offset; IMU is not used by this controller."""
        stamp = self.image_stamp
        stamp_s = None if stamp is None else stamp.sec + stamp.nanosec * 1e-9
        now_s = self.get_clock().now().nanoseconds * 1e-9
        # Reject future/nonfinite timestamps and reset after a clock reversal.
        if stamp_s is not None and (not math.isfinite(stamp_s) or stamp_s > now_s):
            self.estimator.reset()
            return None
        if (self.estimator.last_used_stamp_s is not None and
                now_s < self.estimator.last_used_stamp_s):
            self.estimator.reset()
            return None
        if (stamp_s is not None and self.estimator.last_used_stamp_s is not None
                and stamp_s < self.estimator.last_used_stamp_s):
            return None
        return self.estimator.update(
            observation.error if observation is not None else None,
            observation.found if observation is not None else False,
            stamp_s, now_s)

    def choose_action(self, estimate):
        """Pass the estimate to control; missing road or blue boundary stops."""
        return estimate

    def compute_motion(self, action):
        return self.controller.compute(
            action,
            road_found=self.observation is not None and self.observation.found,
            boundary_warning=self.observation is not None and self.observation.blue_warning)

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

"""Camera navigation pipeline with route recording and replay advice.

This node intentionally publishes zero motion. Route replay is advisory only;
it does not steer the robot or bypass the safety node.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CompressedImage, Imu
from course_robot_interfaces.msg import MotionRequest
from a2r_robot_interfaces.msg import EncoderState

from .route_memory import DeadReckoner, RouteMemory, RouteAdvisor, NO_ADVICE
from .perception import Perception
from .estimation import RoadEstimator


class Navigation(Node):
    def __init__(self):
        super().__init__("navigation")

        self.target_speed = float(self.declare_parameter(
            "requested_forward_speed_m_s", 0.05).value)
        self.route_mode = str(self.declare_parameter("route_mode", "off").value)
        self.route_file = str(self.declare_parameter(
            "route_file", "~/.ros/team_navigation/routes/test_route.json").value)
        self.route_overwrite = bool(self.declare_parameter(
            "route_overwrite", False).value)
        self.route_save_interval = float(self.declare_parameter(
            "route_save_interval_s", 1.0).value)
        self.route_spacing = float(self.declare_parameter(
            "route_spacing_m", 0.05).value)
        self.route_cell = float(self.declare_parameter(
            "route_cell_m", 0.10).value)
        self.route_lookahead = float(self.declare_parameter(
            "route_lookahead_m", 0.50).value)
        self.route_match_window = float(self.declare_parameter(
            "route_match_window_m", 0.40).value)
        self.route_max_match = float(self.declare_parameter(
            "route_max_match_m", 0.30).value)
        self.max_imu_age = float(self.declare_parameter(
            "max_imu_age_s", 0.25).value)
        self.max_encoder_age = float(self.declare_parameter(
            "max_encoder_age_s", 0.25).value)

        if self.route_mode not in ("off", "record", "replay"):
            raise ValueError("route_mode must be 'off', 'record', or 'replay'")
        if not all(math.isfinite(v) and v > 0 for v in (
                self.route_save_interval, self.route_spacing, self.route_cell,
                self.route_lookahead, self.route_match_window,
                self.route_max_match, self.max_imu_age, self.max_encoder_age)):
            raise ValueError("Route distances, save interval, and sensor age limits must be positive")
        if not math.isfinite(self.target_speed) or self.target_speed < 0:
            raise ValueError("requested_forward_speed_m_s must be finite and nonnegative")

        self.route_path = Path(self.route_file).expanduser()
        self.last_route_save_t = None
        self.route_recording_failed = False
        self.route_replay_failed = False
        self.last_route_event = None
        self.last_advice_log = None

        self.image = None
        self.image_stamp = None
        self.imu = None
        self.encoder = None
        self.imu_stamp = None
        self.encoder_stamp = None
        self.last_encoder_stamp = None
        self.hardware_session = None

        self.pose = DeadReckoner()
        self.route_memory = RouteMemory(
            spacing_m=self.route_spacing, cell_m=self.route_cell)
        self.route_advisor = None

        if self.route_mode == "record":
            if self.route_path.exists() and not self.route_overwrite:
                raise FileExistsError(
                    f"Route already exists: {self.route_path}. Choose another "
                    "route_file or explicitly set route_overwrite:=true.")
        elif self.route_mode == "replay":
            reference = RouteMemory.load(self.route_path)
            if len(reference) < 3:
                raise ValueError("Reference route needs at least three recorded points")
            self.route_advisor = RouteAdvisor(
                reference,
                lookahead_m=self.route_lookahead,
                window_m=self.route_match_window,
                max_match_m=self.route_max_match,
            )
            self.get_logger().info(f"Loaded reference route: {len(reference)} points")

        self.perception = Perception()
        self.estimator = RoadEstimator(
            smoothing_alpha=0.2, max_image_age_s=0.25,
            road_lost_timeout_s=0.5, max_jump=0.3,
            outlier_accept_count=3)
        self.observation = None
        self.observation_stamp = None
        self.last_estimator_status = None

        self.create_subscription(
            CompressedImage, "camera/image_raw/compressed",
            self.image_cb, qos_profile_sensor_data)
        self.create_subscription(
            Imu, "imu/data", self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(
            EncoderState, "encoders", self.encoder_cb,
            qos_profile_sensor_data)

        self.publisher = self.create_publisher(
            MotionRequest, "navigation/request",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            f"Navigation started (route_mode={self.route_mode}); motion is disabled")

    @staticmethod
    def stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0

    def ros_seconds(self):
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def _fresh(self, stamp, max_age):
        if stamp is None:
            return False
        age = self.ros_seconds() - stamp
        return math.isfinite(age) and 0.0 <= age <= max_age

    def image_cb(self, msg):
        try:
            jpeg = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            frame = cv2.imdecode(jpeg, cv2.IMREAD_REDUCED_COLOR_2)
            if frame is None:
                raise ValueError("OpenCV could not decode JPEG frame")
            self.image = frame
            self.image_stamp = msg.header.stamp
        except Exception as exc:
            self.image = self.image_stamp = None
            self.observation = self.observation_stamp = None
            self.get_logger().error(f"Camera decode failed: {exc}")

    def imu_cb(self, msg):
        stamp = self.stamp_to_seconds(msg.header.stamp)
        if not math.isfinite(stamp):
            return
        if self.imu_stamp is not None and stamp <= self.imu_stamp:
            return
        self.imu = msg
        self.imu_stamp = stamp
        self.pose.update_gyro(float(msg.angular_velocity.z), stamp)

    def encoder_cb(self, msg):
        stamp = self.stamp_to_seconds(msg.header.stamp)
        if not math.isfinite(stamp):
            return

        if self.hardware_session != msg.hardware_session:
            if self.hardware_session is not None:
                self.get_logger().warning(
                    "Hardware session changed; resetting odometry and disabling this route run")
                self.pose.reset()
                self.last_encoder_stamp = None
                self.imu_stamp = None
                if self.route_mode == "record":
                    self.route_recording_failed = True
                elif self.route_mode == "replay":
                    self.route_replay_failed = True
                    if self.route_advisor is not None:
                        self.route_advisor.reset()
            self.hardware_session = msg.hardware_session

        if self.encoder_stamp is not None and stamp <= self.encoder_stamp:
            return
        self.encoder = msg
        self.encoder_stamp = stamp

        if not msg.calibrated:
            return
        left, right = float(msg.left_delta_m), float(msg.right_delta_m)
        if not (math.isfinite(left) and math.isfinite(right)):
            return
        self.last_encoder_stamp = stamp
        self.pose.update_distance((left + right) / 2.0)

    def process_image(self, image):
        if image is None or self.image_stamp is None:
            return None
        key = (self.image_stamp.sec, self.image_stamp.nanosec)
        if key == self.observation_stamp:
            return self.observation
        result = self.perception.process(image)
        self.observation = result
        self.observation_stamp = key
        return result

    def update_estimate(self, observation):
        now = self.ros_seconds()
        if observation is None or self.image_stamp is None:
            estimate = self.estimator.update(
                offset=None, found=False, image_stamp_s=None, now_s=now)
        else:
            estimate = self.estimator.update(
                offset=observation.error, found=observation.found,
                image_stamp_s=self.stamp_to_seconds(self.image_stamp),
                now_s=now)
        if estimate.status != self.last_estimator_status:
            self.get_logger().info(
                f"Road estimate: status={estimate.status}, valid={estimate.valid}, "
                f"offset={estimate.offset}, image_age={estimate.image_age_s:.3f}s, "
                f"road_age={estimate.road_age_s:.3f}s")
            self.last_estimator_status = estimate.status
        return estimate

    def _record_transition_event(self, estimate):
        if self.route_mode != "record" or self.route_recording_failed:
            return
        current = bool(estimate.valid)
        if self.last_route_event is None:
            self.last_route_event = current
            return
        if current != self.last_route_event:
            kind = "road_found" if current else "road_lost"
            self.route_memory.mark_event(
                kind, self.pose, self.ros_seconds(),
                status=str(estimate.status))
            self.last_route_event = current

    def record_route(self, estimate):
        if self.route_mode != "record" or self.route_recording_failed:
            return
        if (self.encoder is None or not self.encoder.calibrated
                or not self._fresh(self.encoder_stamp, self.max_encoder_age)
                or not self._fresh(self.imu_stamp, self.max_imu_age)):
            return

        obs = self.observation
        speed_values = (float(self.encoder.left_speed_m_s),
                        float(self.encoder.right_speed_m_s))
        speed = sum(speed_values) / 2.0 if all(
            math.isfinite(v) for v in speed_values) else None
        self.route_memory.record(
            pose=self.pose, t=self.ros_seconds(),
            offset=estimate.offset, valid=estimate.valid,
            yellow_px=obs.yellow_pixels if obs is not None else None,
            blue_left=obs.blue_left if obs is not None else None,
            blue_right=obs.blue_right if obs is not None else None,
            v=speed, omega=float(self.imu.angular_velocity.z) if self.imu else None)

    def save_route_if_needed(self, force=False):
        if self.route_mode != "record" or self.route_recording_failed:
            return
        now = self.ros_seconds()
        if (not force and self.last_route_save_t is not None
                and now - self.last_route_save_t < self.route_save_interval):
            return
        if len(self.route_memory) == 0:
            return
        self.route_path.parent.mkdir(parents=True, exist_ok=True)
        self.route_memory.save(self.route_path)
        self.last_route_save_t = now
        self.get_logger().info(
            f"Saved route: {len(self.route_memory)} points to {self.route_path}")

    def choose_action(self, estimate):
        advice = NO_ADVICE
        if (self.route_mode == "replay" and self.route_advisor is not None
                and not self.route_replay_failed
                and self.encoder is not None and self.encoder.calibrated
                and self._fresh(self.encoder_stamp, self.max_encoder_age)
                and self._fresh(self.imu_stamp, self.max_imu_age)):
            advice = self.route_advisor.advice(
                self.pose.x, self.pose.y, self.pose.s)
        if advice != self.last_advice_log:
            if advice.confident:
                self.get_logger().info(
                    f"Route advice: speed_scale={advice.speed_scale:.2f}, "
                    f"curvature={advice.curvature_ahead:.2f}, "
                    f"hazard={advice.hazard_ahead}, "
                    f"match_error={advice.match_dist:.2f}m")
            self.last_advice_log = advice
        return {"estimate": estimate, "advice": advice}

    def compute_motion(self, action):
        """Intentionally stationary until control is separately reviewed."""
        return 0.0, 0.0

    def tick(self):
        forward, yaw = 0.0, 0.0
        try:
            observation = self.process_image(self.image)
            estimate = self.update_estimate(observation)
            self._record_transition_event(estimate)
            self.record_route(estimate)
            self.save_route_if_needed()
            action = self.choose_action(estimate)
            forward, yaw = self.compute_motion(action)
            if not (math.isfinite(forward) and math.isfinite(yaw)):
                raise ValueError("Controller returned a nonfinite command")
        except Exception as exc:
            forward, yaw = 0.0, 0.0
            self.get_logger().error(
                f"Navigation failed; requesting zero motion: {exc}")

        msg = MotionRequest()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.forward_speed_m_s = float(forward)
        msg.yaw_rate_rad_s = float(yaw)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = None
    try:
        node = Navigation()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.save_route_if_needed(force=True)
            except Exception as exc:
                node.get_logger().error(f"Could not save route during shutdown: {exc}")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

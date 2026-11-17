"""Receive sensors and request motion through the supplied safety node.

The supplied hooks return no action. Implement perception and control here
or distribute those responsibilities across additional team nodes.
"""
# Course attribution: this file began as the navigation starter prepared by the
# A2R Lab for COSC 81/281, Fall 2026, released under the MIT license (see
# robot/NOTICE.md and LICENSE).
#
# Team The Perceptrons: drive-basic wiring. Each hook hands off to a teammate's
# part where one exists, so each part can be swapped or improved on its own:
#   process_image   -> perception.py  (Tanner)  yellow road + blue boundary
#   update_estimate -> estimation.py  (Michael) smoothing, stale/glitch checks
#   choose_action   -> basic planning (placeholder; replace with planning's version)
#   compute_motion  -> basic P control (placeholder; replace with control's version)
#
# Sign conventions:
#   perception error / estimate offset: -1 = road far LEFT, +1 = road far RIGHT
#   yaw rate: positive = turn LEFT (ROS convention)
#   so road on the right (offset > 0) -> right turn (yaw < 0): yaw = -gain * offset

import csv
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, Imu, CompressedImage
from cv_bridge import CvBridge
from course_robot_interfaces.msg import MotionRequest
from a2r_robot_interfaces.msg import EncoderState

from team_navigation.perception import Perception, load_config
from team_navigation.estimation import RoadEstimator
from team_navigation.route_memory import DeadReckoner, RouteMemory, RouteAdvisor, NO_ADVICE


# Every tunable number, with defaults. config/navigation.yaml overrides these,
# and `ros2 param set "/$ROBOT_NAMESPACE/navigation" NAME VALUE` changes them
# live (they're re-read every tick). Floats must keep a decimal point.
PARAMETER_DEFAULTS = {
    # --- control ---
    'requested_forward_speed_m_s': 0.30,   # starter's name, kept. Above 0.15 needs Medium (press Y).
    # Speed-scaled steering: yaw = -gain_per_m * forward_speed * offset, so the same
    # gain follows the same curves at any speed (robot-5 tests: 0.8 rad/s at 0.10 m/s,
    # 2.4 at 0.30 and 3.6 at 0.45 all tracked curves -> ~8 per metre).
    # Set to 0.0 to fall back to the fixed steering_gain_rad_s below.
    'steering_gain_per_m': 8.0,
    'steering_gain_rad_s': 0.8,            # legacy fixed gain: rad/s per unit offset (offset -1..+1)
    'max_yaw_rate_rad_s': 2.4,
    # --- perception look-ahead (fractions of the bottom-third ROI; 0 = far edge, 1 = nearest row) ---
    # Lower band = aim at road closer to the robot = turn later / cut corners less.
    # Faster driving needs to look farther ahead to react in time, so the band slides
    # linearly from the slow band (at/below lookahead_slow_speed_m_s) to the fast band
    # (at/above lookahead_fast_speed_m_s). Set the fast band equal to the slow one to disable.
    'lookahead_band_top': 0.45,            # slow band; perception.py default 0.25
    'lookahead_band_bottom': 0.85,         # slow band; perception.py default 0.55
    'lookahead_fast_band_top': 0.15,
    'lookahead_fast_band_bottom': 0.55,
    'lookahead_slow_speed_m_s': 0.30,
    'lookahead_fast_speed_m_s': 0.80,
    'lookahead_min_coverage': 0.3,         # perception.py default 0.5; lower = keep following a road end longer
    'use_compressed_image': True,          # JPEG topic arrives far fresher than raw over DDS (read at startup)
    # --- planning ---
    'follow_while_holding': True,          # keep following the estimator's held offset while the road
    'holding_speed_scale': 0.6,            # is briefly missing (< road_lost_timeout_s), at this speed fraction
    'blue_slowdown_factor': 1.0,   # multiply speed by this while blue boundary is close (1.0 = off)
    'branch_choice': 'all',        # 'all' = center of all yellow (perception default);
                                   # 'left' / 'right' / 'largest' = follow only that yellow blob (for forks)
    'min_branch_pixels': 60,       # blobs smaller than this (in perception's 320x240 image) are ignored
    # --- perception colors ---
    'hsv_config_path': '~/hsv_config.json',   # same file track_test's sliders save to
    # --- estimation (passed to RoadEstimator) ---
    'smoothing_alpha': 0.2,
    'max_image_age_s': 0.25,
    'road_lost_timeout_s': 0.5,
    'max_jump': 0.3,
    'outlier_accept_count': 3,
    # --- route memory (Guillermo's route_memory.py); mode/file/resolution read at startup ---
    'route_mode': 'off',                   # 'off' | 'record' (save a route) | 'replay' (use its advice)
    'route_file': '~/.ros/team_navigation/routes/test_route.json',
    'route_overwrite': False,
    'route_save_interval_s': 1.0,
    'route_spacing_m': 0.05,
    'route_cell_m': 0.10,
    'route_lookahead_m': 0.50,
    'route_match_window_m': 0.40,
    'route_max_match_m': 0.30,
    'route_advice_weight': 1.0,            # live: 0 = log advice only, 1 = apply its speed scale fully
    'max_imu_age_s': 0.25,
    'max_encoder_age_s': 0.25,
    # --- logging ---
    'log_period_s': 0.25,
    'csv_logging': True,
    'log_dir': '~/nav_runs',
}


@dataclass
class Observation:
    """process_image's output: perception's result plus the image capture time."""
    result: object        # perception.PerceptionResult
    stamp_s: float


@dataclass
class Action:
    """choose_action's output."""
    mode: str             # 'follow' or 'stop'
    reason: str
    offset: float = 0.0
    speed_scale: float = 1.0


class Navigation(Node):
    def __init__(self):
        super().__init__('navigation')
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        self.read_parameters()

        self.bridge = CvBridge()
        self.image = self.image_stamp = self.imu = None
        self.jpeg = None              # latest compressed frame, decoded lazily in tick()
        self.previous_error = None
        self.last_observation = None

        # Perception colors: defaults, overridden by the JSON file track_test saves.
        cfg_path = self.cfg['hsv_config_path']
        self.perception = Perception(load_config(cfg_path))
        self.get_logger().info(f'Perception HSV config: {self.perception.cfg} '
                               f'(from {cfg_path} if it exists, else defaults)')
        self.estimator = RoadEstimator()
        self.apply_estimator_params()
        self.setup_route()

        self.last_print_s = -math.inf
        self.last_flush_s = -math.inf
        self.csv_file = self.csv_writer = None
        self.open_log()

        if self.cfg['use_compressed_image']:
            # Only JPEG crosses DDS; raw frames often arrived >0.25 s old (stale-image stops).
            self.create_subscription(CompressedImage, 'camera/image_raw/compressed',
                                     self.compressed_cb, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, 'camera/image_raw', self.image_cb, qos_profile_sensor_data)
        self.get_logger().info('Camera topic: ' + ('camera/image_raw/compressed'
                               if self.cfg['use_compressed_image'] else 'camera/image_raw'))
        self.create_subscription(Imu, 'imu/data', self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(EncoderState, 'encoders', self.encoder_cb, qos_profile_sensor_data)
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

    def compressed_cb(self, msg):
        # Keep only the newest JPEG; decoding happens once per control tick.
        self.jpeg = msg

    def decode_latest(self):
        """Decode the newest compressed frame (if any) into self.image / self.image_stamp."""
        msg, self.jpeg = self.jpeg, None
        if msg is None:
            return
        # Half-resolution decode (Guillermo's trick): perception shrinks to 320x240
        # anyway, and this roughly halves decode time on the Jetson.
        frame = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        if frame is None:
            self.get_logger().error('JPEG decode failed; keeping previous frame')
            return
        self.image, self.image_stamp = frame, msg.header.stamp

    def imu_cb(self, msg):
        self.imu = msg
        # Gyro heading for route dead reckoning (ignored if the stamp is unusable/out of order).
        try:
            stamp = self.stamp_to_seconds(msg.header.stamp)
            if math.isfinite(stamp) and (self.imu_stamp is None or stamp > self.imu_stamp):
                self.imu_stamp = stamp
                self.pose.update_gyro(float(msg.angular_velocity.z), stamp)
        except (AttributeError, TypeError, ValueError):
            pass

    def encoder_cb(self, msg):
        """Wheel travel for route dead reckoning (ported from Guillermo's node)."""
        try:
            stamp = self.stamp_to_seconds(msg.header.stamp)
        except (AttributeError, TypeError, ValueError):
            return
        if not math.isfinite(stamp):
            return
        if self.hardware_session != msg.hardware_session:
            if self.hardware_session is not None:
                self.get_logger().warning(
                    'Hardware session changed; resetting odometry and disabling this route run')
                self.pose.reset()
                self.imu_stamp = None
                if self.route_mode != 'off':
                    self.route_failed = True
                if self.route_advisor is not None:
                    self.route_advisor.reset()
            self.hardware_session = msg.hardware_session
        if self.encoder_stamp is not None and stamp <= self.encoder_stamp:
            return
        self.encoder, self.encoder_stamp = msg, stamp
        if not msg.calibrated:
            return
        left, right = float(msg.left_delta_m), float(msg.right_delta_m)
        if math.isfinite(left) and math.isfinite(right):
            self.pose.update_distance((left + right) / 2.0)

    # ------------------------------------------------------------ route memory
    @staticmethod
    def stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _fresh(self, stamp, max_age):
        if stamp is None:
            return False
        age = self.now_s() - stamp
        return math.isfinite(age) and 0.0 <= age <= max_age

    def setup_route(self):
        """Record/replay set-up from Guillermo's route_memory (startup-only parameters)."""
        c = self.cfg
        self.route_mode = str(c['route_mode'])
        if self.route_mode not in ('off', 'record', 'replay'):
            raise ValueError("route_mode must be 'off', 'record', or 'replay'")
        positive = ('route_save_interval_s', 'route_spacing_m', 'route_cell_m', 'route_lookahead_m',
                    'route_match_window_m', 'route_max_match_m', 'max_imu_age_s', 'max_encoder_age_s')
        if not all(math.isfinite(c[k]) and c[k] > 0 for k in positive):
            raise ValueError('Route distances, save interval, and sensor age limits must be positive')
        self.route_path = Path(c['route_file']).expanduser()
        self.pose = DeadReckoner()
        self.route_memory = RouteMemory(spacing_m=c['route_spacing_m'], cell_m=c['route_cell_m'])
        self.route_advisor = None
        self.route_failed = False
        self.encoder = self.encoder_stamp = self.imu_stamp = self.hardware_session = None
        self.last_route_event = self.last_route_save_t = self.last_advice_log = None
        self.advice = NO_ADVICE
        if self.route_mode == 'record':
            if self.route_path.exists() and not c['route_overwrite']:
                raise FileExistsError(f'Route already exists: {self.route_path}. Choose another '
                                      'route_file or explicitly set route_overwrite:=true.')
        elif self.route_mode == 'replay':
            reference = RouteMemory.load(self.route_path)
            if len(reference) < 3:
                raise ValueError('Reference route needs at least three recorded points')
            self.route_advisor = RouteAdvisor(reference, lookahead_m=c['route_lookahead_m'],
                                              window_m=c['route_match_window_m'],
                                              max_match_m=c['route_max_match_m'])
            self.get_logger().info(f'Loaded reference route: {len(reference)} points')
        self.get_logger().info(f'Route mode: {self.route_mode} ({self.route_path})')

    def sensors_fresh(self):
        return (self.encoder is not None and self.encoder.calibrated
                and self._fresh(self.encoder_stamp, self.cfg['max_encoder_age_s'])
                and self._fresh(self.imu_stamp, self.cfg['max_imu_age_s']))

    def route_step(self, estimate):
        """Record breadcrumbs/events and save (record), or refresh advice (replay)."""
        self.advice = NO_ADVICE
        if self.route_mode == 'off' or self.route_failed or estimate is None:
            return
        now = self.now_s()
        if self.route_mode == 'record':
            valid = bool(estimate.valid)
            if self.last_route_event is not None and valid != self.last_route_event:
                self.route_memory.mark_event('road_found' if valid else 'road_lost', self.pose, now,
                                             status=str(estimate.status))
            self.last_route_event = valid
            if self.sensors_fresh():
                r = self.last_observation.result if self.last_observation else None
                speeds = (float(self.encoder.left_speed_m_s), float(self.encoder.right_speed_m_s))
                self.route_memory.record(
                    pose=self.pose, t=now, offset=estimate.offset, valid=estimate.valid,
                    yellow_px=r.yellow_pixels if r else None,
                    blue_left=r.blue_left if r else None, blue_right=r.blue_right if r else None,
                    v=sum(speeds) / 2 if all(math.isfinite(s) for s in speeds) else None,
                    omega=float(self.imu.angular_velocity.z) if self.imu is not None else None)
            self.save_route_if_needed()
        elif self.route_advisor is not None and self.sensors_fresh():
            self.advice = self.route_advisor.advice(self.pose.x, self.pose.y, self.pose.s)
            if self.advice.confident and self.advice != self.last_advice_log:
                self.get_logger().info(
                    f'Route advice: speed_scale={self.advice.speed_scale:.2f}, '
                    f'curvature={self.advice.curvature_ahead:.2f}, hazard={self.advice.hazard_ahead}, '
                    f'match_error={self.advice.match_dist:.2f}m', throttle_duration_sec=1.0)
            self.last_advice_log = self.advice

    def save_route_if_needed(self, force=False):
        if self.route_mode != 'record' or self.route_failed or len(self.route_memory) == 0:
            return
        now = self.now_s()
        if (not force and self.last_route_save_t is not None
                and now - self.last_route_save_t < self.cfg['route_save_interval_s']):
            return
        self.route_path.parent.mkdir(parents=True, exist_ok=True)
        self.route_memory.save(self.route_path)
        self.last_route_save_t = now

    def route_speed_factor(self):
        """Blend replay advice into speed: 1 - weight * (1 - advice), 1.0 when not confident."""
        if not self.advice.confident:
            return 1.0
        w = max(0.0, min(1.0, float(self.cfg['route_advice_weight'])))
        return 1.0 - w * (1.0 - self.advice.speed_scale)

    # ------------------------------------------------------------ helpers
    def read_parameters(self):
        self.cfg = {name: self.get_parameter(name).value for name in PARAMETER_DEFAULTS}

    def apply_estimator_params(self):
        for name in ('smoothing_alpha', 'max_image_age_s', 'road_lost_timeout_s',
                     'max_jump', 'outlier_accept_count'):
            setattr(self.estimator, name, self.cfg[name])

    def lookahead_band(self):
        """(top, bottom) band for the current requested speed: slow band at low
        speed, fast (farther) band at high speed, linear blend in between."""
        c = self.cfg
        slow_v, fast_v = c['lookahead_slow_speed_m_s'], c['lookahead_fast_speed_m_s']
        if not fast_v > slow_v:
            raise ValueError(f'lookahead_fast_speed_m_s ({fast_v}) must exceed slow ({slow_v})')
        t = (c['requested_forward_speed_m_s'] - slow_v) / (fast_v - slow_v)
        t = max(0.0, min(1.0, t))
        top = c['lookahead_band_top'] + t * (c['lookahead_fast_band_top'] - c['lookahead_band_top'])
        bottom = c['lookahead_band_bottom'] + t * (c['lookahead_fast_band_bottom'] - c['lookahead_band_bottom'])
        return top, bottom

    def apply_perception_params(self):
        top, bottom = self.lookahead_band()
        coverage = self.cfg['lookahead_min_coverage']
        if not (0.0 <= top < bottom <= 1.0 and 0.0 < coverage <= 1.0):
            raise ValueError(f'invalid look-ahead band top={top} bottom={bottom} coverage={coverage}')
        self.perception.band_top = top
        self.perception.band_bottom = bottom
        self.perception.min_coverage = coverage

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ hook 1: perception
    def process_image(self, image):
        """Tanner's perception on the newest frame; None if no image yet."""
        if image is None or self.image_stamp is None:
            return None
        stamp_s = self.image_stamp.sec + self.image_stamp.nanosec * 1e-9
        result = self.perception.process(image)
        if self.cfg['branch_choice'] != 'all' and result.found:
            result = self.pick_branch(result)
        return Observation(result, stamp_s)

    def pick_branch(self, r):
        """At a fork, follow ONE yellow blob instead of the center of all yellow.

        Perception averages every yellow pixel in its region, so with two
        branches in view it would aim at the gap between them. Here we split
        the yellow mask into separate blobs (connected components) and keep
        only the leftmost / rightmost / largest one.
        """
        roi = r.yellow_mask[r.roi_top:, :]
        count, _, stats, centroids = cv2.connectedComponentsWithStats(roi)
        # label 0 is the background; keep blobs above the size floor
        blobs = [(stats[i, cv2.CC_STAT_AREA], centroids[i][0])
                 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= self.cfg['min_branch_pixels']]
        if len(blobs) < 2:
            return r                                  # no fork in view: keep perception's answer
        choice = self.cfg['branch_choice']
        if choice == 'left':
            _, cx = min(blobs, key=lambda b: b[1])
        elif choice == 'right':
            _, cx = max(blobs, key=lambda b: b[1])
        else:                                         # 'largest'
            _, cx = max(blobs, key=lambda b: b[0])
        w = roi.shape[1]
        error = float(np.clip((cx - w / 2) / (w / 2), -1.0, 1.0))
        return replace(r, error=error, cx=float(cx))

    # ------------------------------------------------------------ hook 2: estimation
    def update_estimate(self, observation, imu):
        """Michael's RoadEstimator (IMU unused)."""
        now_s = self.now_s()
        if observation is None:
            return self.estimator.update(None, False, None, now_s)
        r = observation.result
        return self.estimator.update(r.error, r.found, observation.stamp_s, now_s)

    # ------------------------------------------------------------ hook 3: planning (basic)
    def choose_action(self, estimate):
        """Follow while the estimate is valid; otherwise stop. Slow down near blue.

        When the road is briefly missing (estimator status 'holding', still
        valid until road_lost_timeout_s), keep following the held offset at
        holding_speed_scale instead of stopping on the first empty frame.
        """
        if estimate is None or not estimate.valid:
            status = estimate.status if estimate is not None else 'no_estimate'
            return Action('stop', status)
        seen = self.last_observation is not None and self.last_observation.result.found
        scale = 1.0
        if not seen:
            if not (self.cfg['follow_while_holding'] and estimate.status == 'holding'):
                return Action('stop', 'no_road_ahead')
            scale = self.cfg['holding_speed_scale']
        if self.last_observation is not None and self.last_observation.result.blue_warning:
            scale *= self.cfg['blue_slowdown_factor']
        scale *= self.route_speed_factor()       # replay advice: slow before known curves/hazards
        return Action('follow', estimate.status, estimate.offset, scale)

    # ------------------------------------------------------------ hook 4: control (basic)
    def compute_motion(self, action):
        """Return (forward m/s, left-turn rad/s); no action means zero motion.

        Implement your controller here. A positive yaw request turns left.
        For student-determined separation b (metres), safety converts to treads
        v - b*w/2 and v + b*w/2. Y selects the global tread-speed limit:
        Low 0.15, Medium 0.60, or High 1.40 m/s. If either exceeds that limit,
        it scales both together, preserving their ratio without disarming.
        Invalid or stale requests still stop and disarm. See the README.

        Proportional control on the road offset. With steering_gain_per_m > 0
        the turn rate scales with forward speed (a curvature command), so one
        gain tracks the same curves at any speed:
            yaw = -steering_gain_per_m * forward * offset
        e.g. gain 8, 0.30 m/s, offset +0.5 -> yaw -1.2 rad/s (right).
        With steering_gain_per_m == 0 it falls back to the fixed gain:
            yaw = -steering_gain_rad_s * offset
        """
        if action is None or action.mode != 'follow':
            return 0., 0.
        forward = self.cfg['requested_forward_speed_m_s'] * action.speed_scale
        if self.cfg['steering_gain_per_m'] > 0:
            yaw = -self.cfg['steering_gain_per_m'] * forward * action.offset
        else:
            yaw = -self.cfg['steering_gain_rad_s'] * action.offset
        limit = self.cfg['max_yaw_rate_rad_s']
        return forward, max(-limit, min(limit, yaw))

    # ------------------------------------------------------------ main loop
    def tick(self):
        forward, yaw = 0., 0.
        observation = estimate = action = None
        self.last_observation = None
        error_text = ''
        try:
            self.read_parameters()
            self.apply_estimator_params()
            self.apply_perception_params()
            self.decode_latest()
            observation = self.process_image(self.image)
            self.last_observation = observation
            estimate = self.update_estimate(observation, self.imu)
            try:
                self.route_step(estimate)
            except Exception as exc:          # route features must never stop driving
                self.route_failed, self.advice = True, NO_ADVICE
                self.get_logger().error(f'Route memory disabled after error: {exc}')
            action = self.choose_action(estimate)
            forward, yaw = self.compute_motion(action)
            if not all(math.isfinite(v) for v in (forward, yaw)):
                raise ValueError('controller returned a nonfinite command')
            self.previous_error = None
        except Exception as exc:
            forward, yaw = 0., 0.
            error = str(exc)
            error_text = error
            if error != self.previous_error:
                self.get_logger().error(f'Navigation failed; requesting zero motion: {error}')
                self.previous_error = error
        msg = MotionRequest()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.forward_speed_m_s, msg.yaw_rate_rad_s = float(forward), float(yaw)
        self.publisher.publish(msg)
        self.record(observation, estimate, action, forward, yaw, error_text)

    # ------------------------------------------------------------ logging
    COLUMNS = ['ros_time_s', 'image_age_s', 'found', 'raw_error', 'yellow_pixels',
               'blue_left', 'blue_right', 'estimate', 'valid', 'est_status',
               'action', 'forward_m_s', 'yaw_rad_s', 'error', 'route_factor', 'route_s']

    def open_log(self):
        """One CSV per launch in ~/nav_runs/run_<date>_<time>/ (outside the repo)."""
        if not self.cfg['csv_logging']:
            return
        try:
            run_dir = os.path.join(os.path.expanduser(self.cfg['log_dir']),
                                   datetime.now().strftime('run_%Y%m%d_%H%M%S'))
            os.makedirs(run_dir, exist_ok=True)
            path = os.path.join(run_dir, 'nav_log.csv')
            self.csv_file = open(path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.COLUMNS)
            self.get_logger().info(f'CSV log: {path}')
        except Exception as exc:
            self.csv_file = self.csv_writer = None
            self.get_logger().error(f'CSV logging off: {exc}')

    def record(self, observation, estimate, action, forward, yaw, error_text):
        try:
            now_s = self.now_s()
            r = observation.result if observation else None
            age = estimate.image_age_s if estimate else math.nan
            raw = r.error if (r and r.found) else math.nan
            est = estimate.offset if (estimate and estimate.offset is not None) else math.nan
            row = [f'{now_s:.4f}', f'{age:.4f}', int(bool(r and r.found)), f'{raw:+.4f}',
                   r.yellow_pixels if r else 0,
                   f'{r.blue_left:.3f}' if r else '', f'{r.blue_right:.3f}' if r else '',
                   f'{est:+.4f}', int(bool(estimate and estimate.valid)),
                   estimate.status if estimate else '',
                   action.mode if action else 'stop', f'{forward:.4f}', f'{yaw:+.4f}', error_text,
                   f'{self.route_speed_factor():.3f}', f'{self.pose.s:.3f}']
            if self.csv_writer is not None:
                self.csv_writer.writerow(row)
                if now_s - self.last_flush_s >= 1.0:
                    self.csv_file.flush()
                    self.last_flush_s = now_s
            if now_s - self.last_print_s >= self.cfg['log_period_s']:
                self.last_print_s = now_s
                blue = ' BLUE' if (r and r.blue_warning) else ''
                self.get_logger().info(
                    f'{row[10]:6s} | raw {raw:+.3f} | est {est:+.3f} valid {row[8]} '
                    f'({row[9]}) | img age {age:.3f}s | v {forward:.3f} w {yaw:+.3f}{blue}')
        except Exception as exc:
            self.get_logger().warning(f'Logging failed: {exc}', throttle_duration_sec=5.0)

    def close_log(self):
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = self.csv_writer = None


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
                node.get_logger().error(f'Could not save route during shutdown: {exc}')
            node.close_log()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

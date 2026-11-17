"""Spin-in-place track-width check for the A2R RVR.

Publishes yaw-only MotionRequests (forward = 0) while the safety node reports
autonomous mode (operator holding RB). Sequence: settle, left spin, pause,
right spin, stop. Logs gyro z and measured encoder tread speeds, then reports
the effective track width per direction:  b = mean(v_right - v_left) / mean(gyro_z).
Releasing RB (leaving autonomous mode) aborts the test immediately.
"""
import csv
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from course_robot_interfaces.msg import MotionRequest
from a2r_robot_interfaces.msg import EncoderState

RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6     # rad/s commanded
SPIN_S = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0   # seconds per direction
SKIP_S = 1.0                                                # ignore spin-up at start of each spin
PHASES = [('settle', 0.0, 1.0), ('left', RATE, SPIN_S), ('pause', 0.0, 1.5),
          ('right', -RATE, SPIN_S), ('stop', 0.0, 1.0)]
CONTROL_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)


class SpinTest(Node):
    def __init__(self):
        super().__init__('spin_test')
        self.pub = self.create_publisher(MotionRequest, 'navigation/request', CONTROL_QOS)
        self.create_subscription(String, 'safety/status', self.status_cb, 10)
        self.create_subscription(Imu, 'imu/data', self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(EncoderState, 'encoders', self.enc_cb, qos_profile_sensor_data)
        self.autonomous = False
        self.started = None
        self.done = False
        self.gyro = None
        self.rows = []
        self.timer = self.create_timer(0.05, self.tick)
        out = Path.home() / 'alejo_runs'
        out.mkdir(exist_ok=True)
        self.csv_path = out / f'spin_test_{time.strftime("%H%M%S")}.csv'
        self.get_logger().info(f'Ready: hold RB to start (rate {RATE} rad/s, {SPIN_S} s per side)')

    def status_cb(self, msg):
        self.autonomous = 'mode=autonomous' in msg.data

    def imu_cb(self, msg):
        self.gyro = msg.angular_velocity.z

    def enc_cb(self, msg):
        if self.started is None or self.gyro is None:
            return
        phase, t_in = self.phase()
        self.rows.append((time.monotonic() - self.started, phase, t_in, msg.left_speed_m_s,
                          msg.right_speed_m_s, self.gyro, msg.calibrated, msg.hardware_session))

    def phase(self):
        t = time.monotonic() - self.started
        for name, _, dur in PHASES:
            if t < dur:
                return name, t
            t -= dur
        return 'end', t

    def tick(self):
        msg = MotionRequest()
        msg.header.stamp = self.get_clock().now().to_msg()
        yaw = 0.0
        if not self.done:
            if self.started is None and self.autonomous:
                self.started = time.monotonic()
                self.get_logger().info('Autonomous detected: starting spin sequence')
            if self.started is not None:
                if not self.autonomous:
                    self.get_logger().warn('Left autonomous mode (RB released?): aborting')
                    self.finish()
                else:
                    name, _ = self.phase()
                    if name == 'end':
                        self.finish()
                    else:
                        yaw = dict((n, w) for n, w, _ in PHASES)[name]
        msg.forward_speed_m_s = 0.0
        msg.yaw_rate_rad_s = float(yaw)
        self.pub.publish(msg)

    def finish(self):
        if self.done:
            return
        self.done = True
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'phase', 't_in_phase', 'v_left', 'v_right', 'gyro_z', 'calibrated', 'session'])
            w.writerows(self.rows)
        self.get_logger().info(f'Saved {len(self.rows)} samples to {self.csv_path}')
        sessions = {r[7] for r in self.rows}
        if len(sessions) > 1:
            self.get_logger().warn('Hardware session changed during test; results unreliable')
        for side in ('left', 'right'):
            sel = [r for r in self.rows if r[1] == side and r[2] >= SKIP_S
                   and all(math.isfinite(x) for x in (r[3], r[4], r[5]))]
            if len(sel) < 5:
                self.get_logger().warn(f'{side}: too few steady samples ({len(sel)})')
                continue
            dv = sum(r[4] - r[3] for r in sel) / len(sel)
            g = sum(r[5] for r in sel) / len(sel)
            vl = sum(r[3] for r in sel) / len(sel)
            vr = sum(r[4] for r in sel) / len(sel)
            cmd = RATE if side == 'left' else -RATE
            self.get_logger().info(
                f'{side.upper():5s} n={len(sel):3d} cmd={cmd:+.2f} rad/s  gyro={g:+.3f} rad/s  '
                f'v_left={vl:+.3f} v_right={vr:+.3f} m/s  '
                f'effective width={dv / g if abs(g) > 1e-3 else float("nan"):.3f} m  '
                f'(turn achieved {g / cmd * 100:.0f}% of commanded)')


def main():
    rclpy.init()
    node = SpinTest()
    try:
        while rclpy.ok() and not (node.done and node.started is not None):
            rclpy.spin_once(node, timeout_sec=0.05)
        # keep publishing zero briefly so the base sees a clean stop request
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.finish()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

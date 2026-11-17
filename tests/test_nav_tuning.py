"""Look-ahead band, speed-scaled steering, holding follow-through and JPEG input."""
import sys
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robot/src/team_navigation'))
from team_navigation.perception import Perception, centerline_target


def stamp_now(navigation):
    ns = navigation.get_clock().now().nanoseconds
    return types.SimpleNamespace(sec=ns // 10**9, nanosec=ns % 10**9)


def curve_frame():
    """Road that is centred near the robot but bends right farther ahead."""
    frame = np.zeros((240, 320, 3), np.uint8)
    for y in range(160, 240):
        # nearest row (239) centred at x=160; far rows (160) shifted ~70 px right
        x = int(160 + (239 - y) * 0.9)
        frame[y, max(0, x - 15):min(320, x + 15)] = (0, 255, 255)
    return frame


def test_perception_defaults_unchanged():
    p = Perception()
    assert (p.band_top, p.band_bottom, p.min_coverage) == (0.25, 0.55, 0.5)


def test_lower_band_aims_closer_and_turns_less_early():
    frame = curve_frame()
    far = Perception().process(frame)
    near = Perception()
    near.band_top, near.band_bottom = 0.45, 0.85
    near = near.process(frame)
    assert far.found and near.found
    assert near.target_y > far.target_y           # target is lower in the image = closer
    assert 0 < near.error < far.error             # still turns right, but less sharply


def test_lower_coverage_keeps_a_road_end_longer():
    mask = np.zeros((82, 320), np.uint8)
    # ROI 82 rows -> band rows 36..68 (33 rows). Road only in the nearest 14 of them:
    # enough for 30% coverage (needs 10), not for 50% (needs 17).
    mask[55:, 150:170] = 255
    assert centerline_target(mask, 0.45, 0.85, 0.5) is None
    assert centerline_target(mask, 0.45, 0.85, 0.3) is not None


def test_speed_scaled_steering(navigation):
    action = types.SimpleNamespace(mode='follow', offset=0.5, speed_scale=1.0)
    navigation.cfg.update(requested_forward_speed_m_s=0.30, steering_gain_per_m=8.0, max_yaw_rate_rad_s=2.4)
    forward, yaw = navigation.compute_motion(action)
    assert forward == 0.30 and abs(yaw - (-1.2)) < 1e-9
    navigation.cfg.update(requested_forward_speed_m_s=0.10)
    _, yaw_slow = navigation.compute_motion(action)
    assert abs(yaw_slow - (-0.4)) < 1e-9          # same curvature, a third of the rate
    navigation.cfg.update(steering_gain_per_m=0.0, steering_gain_rad_s=0.8)
    _, yaw_legacy = navigation.compute_motion(action)
    assert abs(yaw_legacy - (-0.4)) < 1e-9
    navigation.cfg.update(steering_gain_per_m=8.0, requested_forward_speed_m_s=1.0)
    _, yaw_capped = navigation.compute_motion(action)
    assert yaw_capped == -2.4


def test_follow_through_while_holding(navigation):
    found = types.SimpleNamespace(found=True, blue_warning=False)
    lost = types.SimpleNamespace(found=False, blue_warning=False)
    holding = types.SimpleNamespace(valid=True, status='holding', offset=0.2)
    navigation.last_observation = types.SimpleNamespace(result=lost)
    action = navigation.choose_action(holding)
    assert action.mode == 'follow' and action.speed_scale == navigation.cfg['holding_speed_scale']
    navigation.cfg['follow_while_holding'] = False
    assert navigation.choose_action(holding).mode == 'stop'
    lost_est = types.SimpleNamespace(valid=False, status='lost', offset=0.2)
    navigation.cfg['follow_while_holding'] = True
    assert navigation.choose_action(lost_est).mode == 'stop'
    navigation.last_observation = types.SimpleNamespace(result=found)
    smoothed = types.SimpleNamespace(valid=True, status='smoothed', offset=0.2)
    assert navigation.choose_action(smoothed).speed_scale == 1.0


def test_compressed_frame_is_decoded_on_tick(navigation):
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[160:, 150:170] = (0, 255, 255)
    ok, jpeg = cv2.imencode('.jpg', frame)
    assert ok
    msg = types.SimpleNamespace(data=jpeg.tobytes(), header=types.SimpleNamespace(stamp=stamp_now(navigation)))
    navigation.compressed_cb(msg)
    navigation.decode_latest()
    assert navigation.image is not None and navigation.image.shape == (240, 320, 3)
    assert navigation.jpeg is None
    obs = navigation.process_image(navigation.image)
    assert obs.result.found


def test_invalid_band_rejected(navigation):
    navigation.cfg.update(lookahead_band_top=0.9, lookahead_band_bottom=0.5)
    try:
        navigation.apply_perception_params()
    except ValueError:
        return
    raise AssertionError('inverted band should raise')

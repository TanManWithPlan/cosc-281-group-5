"""Guillermo's route memory wired into the tuned navigation node."""
import json
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robot/src/team_navigation'))
from team_navigation.route_memory import RouteMemory, DeadReckoner, Advice, NO_ADVICE


def stamp(t):
    return types.SimpleNamespace(sec=int(t), nanosec=int((t % 1) * 1e9))


def feed(navigation, t, delta_m=0.02, omega=0.0, session='s1'):
    imu = types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(t)),
                                angular_velocity=types.SimpleNamespace(z=omega))
    enc = types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(t)), hardware_session=session,
                                calibrated=True, left_delta_m=delta_m, right_delta_m=delta_m,
                                left_speed_m_s=0.3, right_speed_m_s=0.3)
    navigation.imu_cb(imu)
    navigation.encoder_cb(enc)


def estimate(valid=True):
    return types.SimpleNamespace(valid=valid, status='smoothed' if valid else 'lost', offset=0.1)


def test_route_off_by_default_and_neutral(navigation):
    assert navigation.route_mode == 'off'
    assert navigation.route_speed_factor() == 1.0
    navigation.route_step(estimate())                 # no-op, no file written
    assert len(navigation.route_memory) == 0


def test_dead_reckoning_from_callbacks(navigation):
    t0 = navigation.now_s()
    for i in range(10):
        feed(navigation, t0 + i * 0.05)
    assert math.isclose(navigation.pose.s, 0.20, rel_tol=1e-6)
    assert math.isclose(navigation.pose.x, 0.20, rel_tol=1e-6)


def test_hardware_session_change_resets_and_disables(navigation):
    navigation.route_mode = 'record'
    t0 = navigation.now_s()
    feed(navigation, t0)
    feed(navigation, t0 + 0.05)
    feed(navigation, t0 + 0.10, session='s2')
    assert navigation.route_failed
    assert navigation.pose.s < 0.03


def test_record_mode_saves_breadcrumbs_and_events(navigation, tmp_path):
    navigation.route_mode = 'record'
    navigation.route_path = tmp_path / 'route.json'
    for i in range(8):
        # stamps just behind "now" so the freshness gates (<= 0.25 s old) accept them
        feed(navigation, navigation.now_s() - 0.001, delta_m=0.06)
        navigation.route_step(estimate(valid=i != 5))
    navigation.save_route_if_needed(force=True)
    saved = RouteMemory.load(navigation.route_path)
    assert len(saved) >= 6
    kinds = [e['kind'] for e in json.loads(navigation.route_path.read_text())['events']]
    assert 'road_lost' in kinds and 'road_found' in kinds


def test_replay_advice_weight_blends_speed(navigation):
    navigation.advice = Advice(0.5, 2.0, False, 1.0, 0.05, True)
    navigation.cfg['route_advice_weight'] = 1.0
    assert navigation.route_speed_factor() == 0.5
    navigation.cfg['route_advice_weight'] = 0.5
    assert navigation.route_speed_factor() == 0.75
    navigation.cfg['route_advice_weight'] = 0.0
    assert navigation.route_speed_factor() == 1.0
    navigation.advice = NO_ADVICE                       # not confident -> neutral
    navigation.cfg['route_advice_weight'] = 1.0
    assert navigation.route_speed_factor() == 1.0


def test_advice_slows_follow_action(navigation):
    navigation.last_observation = types.SimpleNamespace(
        result=types.SimpleNamespace(found=True, blue_warning=False))
    navigation.advice = Advice(0.6, 1.0, False, 1.0, 0.05, True)
    action = navigation.choose_action(types.SimpleNamespace(valid=True, status='smoothed', offset=0.0))
    assert action.mode == 'follow' and math.isclose(action.speed_scale, 0.6)

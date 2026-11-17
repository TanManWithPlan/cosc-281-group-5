"""Exercise the real integrated hooks; no robot or ROS installation needed."""
import types
import numpy as np
import pytest


def deliver(node, x=240, stamp=.99, blue=False, blank=False):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    if not blank: frame[160:, x:x+12] = (0, 255, 255)
    if blue: frame[160:, :60] = (255, 0, 0)
    sec = int(stamp)
    node.image_cb(types.SimpleNamespace(data=frame, header=types.SimpleNamespace(
        stamp=types.SimpleNamespace(sec=sec, nanosec=round((stamp-sec)*1e9)))))
    node.tick()
    return node.messages[-1]


def test_process_image_returns_none_on_missing_image(navigation):
    assert navigation.process_image(None) is None


def test_hook_pipeline(navigation):
    command = deliver(navigation)
    assert navigation.last_observation.result.found
    assert navigation.current_estimate.valid
    assert navigation.choose_action(navigation.current_estimate).mode == 'follow'
    assert command.forward_speed_m_s > 0 and command.yaw_rate_rad_s < 0


def test_frozen_camera_and_recovery(navigation):
    deliver(navigation)
    navigation.now_ns = 1_300_000_000
    navigation.tick()
    assert navigation.messages[-1].forward_speed_m_s == 0
    assert deliver(navigation, stamp=1.3).forward_speed_m_s > 0


def test_blue_stop_and_missing_detection(navigation):
    assert deliver(navigation, blue=True).forward_speed_m_s == 0
    assert deliver(navigation, stamp=1., blank=True).forward_speed_m_s == 0


def test_invalid_live_gain_stops(navigation):
    deliver(navigation)
    navigation.parameters['steering_gain_rad_s'].value = float('nan')
    navigation.tick()
    assert navigation.messages[-1].forward_speed_m_s == 0


@pytest.mark.parametrize('choice,sign', [('left', 1), ('right', -1)])
def test_fork_choice(navigation, choice, sign):
    navigation.parameters['branch_choice'].value = choice
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[160:, 60:75] = frame[160:, 240:255] = (0,255,255)
    navigation.image_cb(types.SimpleNamespace(data=frame, header=types.SimpleNamespace(
        stamp=types.SimpleNamespace(sec=1,nanosec=0))))
    navigation.tick()
    assert navigation.messages[-1].yaw_rate_rad_s * sign > 0

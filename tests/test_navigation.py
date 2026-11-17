import pytest
import glob
import cv2
import types

from pathlib import Path

@pytest.fixture(params=list((Path(__file__).parent / 'data').glob('*.jpg')))
def image(request):
   return request.param

@pytest.fixture
def observation(navigation, image):
    raw_image = cv2.imread(str(image))
    now = navigation.get_clock().now()
    navigation.image_stamp = types.SimpleNamespace(sec=now.nanoseconds // 10**9, nanosec=now.nanoseconds % 10**9)
    return navigation.process_image(raw_image)

@pytest.fixture
def estimate(navigation, observation):
    return navigation.update_estimate(observation, imu=navigation.imu)

@pytest.fixture
def action(navigation, estimate):
   return navigation.choose_action(estimate)

@pytest.fixture
def motion(navigation, action):
   return navigation.compute_motion(action)

# Basic one written by claude
def test_process_image_returns_none_on_missing_image(navigation):
    assert navigation.process_image(None) is None

# Other tests that we need:
# - Test for quality of image processing
# - Not necessarily for this file: robot reaction, etc
# - Edge cases
# - Stale image

#TODO: update these for expected values
def test_process_image(observation):
    print(observation)
    assert observation is not None

def test_update_estimate(estimate):
    print(estimate)
    assert estimate is not None

def test_choose_action(action):
    print(action)
    assert action is not None

def test_ignore_stale_action(navigation, image):
    raw_image = cv2.imread(str(image))
    navigation.image_stamp = types.SimpleNamespace(sec=0, nanosec=0)
    obs = navigation.process_image(raw_image)
    est = navigation.update_estimate(obs, imu=navigation.imu)
    assert est is not None
    act = navigation.choose_action(est)
    assert act.mode == 'stop'
    assert act.reason == 'stale_image'

def test_compute_motion(motion):
    print(motion)
    assert motion is not None


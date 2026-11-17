import pytest
import glob

@pytest.fixture(params=glob.glob("data/*.jpg"))
def image(request):
   return request.param

@pytest.fixture
def observation(navigation, image):
    return navigation.process_image(image)

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

#TODO: update these for expected values
def test_process_image(navigation):
    assert observation is not None

def test_update_estimate(estimate):
    assert estimate is not None

def test_choose_action(action):
    assert action is not None

def test_compute_motion(motion):
    assert motion is not None


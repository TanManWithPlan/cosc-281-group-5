"""Behavioral control and real perception/estimation integration checks."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robot/src/team_navigation'))
from team_navigation.control import RoadController
from team_navigation.estimation import RoadEstimator
from team_navigation.perception import Perception


def estimate(offset=0., age=0., valid=True):
    return SimpleNamespace(offset=offset, image_age_s=age, valid=valid)


def test_steering_direction_and_turn_slowdown():
    c = RoadController()
    straight = c.compute(estimate())
    left = c.compute(estimate(-0.5))
    right = c.compute(estimate(0.5))
    assert straight == (0.05, 0.)
    assert 0 < left[0] == right[0] < straight[0]
    assert left[1] > 0 > right[1]
    assert left[1] == -right[1]
    assert abs(c.compute(estimate(1.))[1]) == c.max_yaw_rate


@pytest.mark.parametrize('e', [None, estimate(valid=False), estimate(age=.251),
    estimate(age=-.01), estimate(age=math.nan), estimate(math.nan),
    estimate(math.inf), estimate(None), estimate(1.1)])
def test_unusable_estimate_stops(e):
    assert RoadController().compute(e) == (0., 0.)


def test_missing_road_and_boundary_stop_immediately():
    c = RoadController()
    assert c.compute(estimate(.5), road_found=False) == (0., 0.)
    assert c.compute(estimate(.5), boundary_warning=True) == (0., 0.)


@pytest.mark.parametrize('kwargs', [{'kp': -1}, {'forward_speed': -1},
    {'max_yaw_rate': 0}, {'turn_slowdown': 2}, {'kp': math.nan},
    {'max_image_age_s': 0}])
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        RoadController(**kwargs)


@pytest.mark.parametrize('x,sign', [(70, 1), (240, -1)])
def test_real_image_to_estimate_to_motion(x, sign):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[160:, x:x+12] = (0, 255, 255)
    p = Perception()
    observation = p.process(frame)
    estimator = RoadEstimator()
    e = estimator.update(observation.error, observation.found, 1., 1.01)
    c = RoadController()
    speed, yaw = c.compute(e, observation.found, observation.blue_warning)
    assert speed > 0 and yaw * sign > 0
    stale = estimator.update(observation.error, True, 1., 1.3)
    assert c.compute(stale) == (0., 0.)
    lost = p.process(np.zeros_like(frame))
    held = estimator.update(lost.error, lost.found, 1.1, 1.1)
    assert held.valid  # estimator holding must not conceal missing detection
    assert c.compute(held, lost.found) == (0., 0.)

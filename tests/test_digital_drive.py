"""Closed-loop regression checks with synthetic camera images."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('digital_drive', Path(__file__).resolve().parents[1]/'tools/digital_drive.py')
sim=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

@pytest.mark.parametrize('name', list(sim.CASES))
def test_image_in_the_loop(name):
    metrics, rows = sim.run_case(name)
    assert metrics['unsafe_motion_samples'] == 0
    assert metrics['final_error_m'] < .04
    if name in ('camera_freeze', 'tape_missing', 'blue_boundary'):
        assert metrics['stopped_samples'] > 0
        assert rows[-1]['forward_m_s'] > 0

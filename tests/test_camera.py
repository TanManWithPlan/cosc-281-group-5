import pytest

def test_process_image_returns_none_on_missing_image(navigation):
    assert navigation.process_image(None) is None

import pytest

# Basic one written by claude
def test_process_image_returns_none_on_missing_image(navigation):
    assert navigation.process_image(None) is None

# Other tests that we need:
# - Test for quality of image processing
# - Not necessarily for this file: robot reaction, etc
# - Edge cases

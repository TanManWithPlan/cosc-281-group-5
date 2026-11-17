"""Sheet-end regressions for the real OpenCV perception pipeline."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robot/src/team_navigation'))
from team_navigation.perception import Perception, centerline_target


class CenterlineTests(unittest.TestCase):
    def test_corner_outside_lookahead_does_not_pull_target(self):
        mask = np.zeros((82, 320), np.uint8)
        mask[:, 110:210] = 255
        mask[:18, 210:315] = 255
        old_center = cv2.moments(mask, binaryImage=True)
        self.assertGreater(old_center['m10'] / old_center['m00'], 175)
        self.assertAlmostEqual(centerline_target(mask)[0], 159.5)

    def test_last_sheet_corner_is_not_a_road_ahead(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.fillPoly(frame, [np.array([[130, 211], [300, 211], [319, 239], [100, 239]])], (0, 255, 255))
        result = Perception().process(frame)
        self.assertGreater(result.yellow_pixels, 200)
        self.assertFalse(result.found)
        self.assertIsNone(result.cx)

    def test_left_and_right_tracks_keep_steering_sign(self):
        for x, sign in ((70, -1), (240, 1)):
            frame = np.zeros((240, 320, 3), np.uint8)
            frame[160:, x:x+12] = (0, 255, 255)
            result = Perception().process(frame)
            self.assertTrue(result.found)
            self.assertGreater(result.error * sign, 0)
            self.assertTrue(175 <= result.target_y <= 205)

    def test_reflection_gap_does_not_split_road_center(self):
        mask = np.zeros((82, 320), np.uint8)
        mask[:, 100:220] = 255
        mask[:, 145:175] = 0
        self.assertAlmostEqual(centerline_target(mask)[0], 159.5)

    def test_empty_image_is_lost(self):
        result = Perception().process(np.zeros((240, 320, 3), np.uint8))
        self.assertFalse(result.found)


if __name__ == '__main__':
    unittest.main()

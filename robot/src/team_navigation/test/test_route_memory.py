import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from team_navigation.route_memory import (
    DeadReckoner, RouteMemory, RouteAdvisor, NO_ADVICE
)


class RouteMemoryTests(unittest.TestCase):
    def setUp(self):
        self.pose = DeadReckoner()

    def test_dead_reckoner_straight_line(self):
        self.pose.update_distance(1.0)
        self.assertAlmostEqual(self.pose.x, 1.0)
        self.assertAlmostEqual(self.pose.y, 0.0)
        self.assertAlmostEqual(self.pose.s, 1.0)

    def test_dead_reckoner_left_turn(self):
        self.pose.update_gyro(1.0, 0.0)
        self.pose.update_gyro(1.0, 0.1)
        self.pose.update_distance(1.0)
        self.assertGreater(self.pose.theta, 0.0)
        self.assertGreater(self.pose.y, 0.0)

    def test_dead_reckoner_rejects_bad_gyro_dt_setting(self):
        with self.assertRaises(ValueError):
            DeadReckoner(0)

    def test_route_memory_spacing(self):
        memory = RouteMemory(spacing_m=0.1)
        self.assertTrue(memory.record(self.pose, 0.0))
        self.pose.update_distance(0.05)
        self.assertFalse(memory.record(self.pose, 0.1))
        self.pose.update_distance(0.06)
        self.assertTrue(memory.record(self.pose, 0.2))

    def test_route_memory_rejects_distance_reset(self):
        memory = RouteMemory(spacing_m=0.01)
        self.pose.update_distance(0.2)
        memory.record(self.pose, 0.0)
        self.pose.s = 0.0
        self.assertFalse(memory.record(self.pose, 1.0))

    def test_visited_cell(self):
        memory = RouteMemory(cell_m=0.1)
        memory.record(self.pose, 0.0)
        self.assertTrue(memory.was_here(0.01, 0.01))
        self.assertFalse(memory.was_here(2.0, 2.0))

    def test_save_load_round_trip(self):
        memory = RouteMemory()
        memory.record(self.pose, 0.0, offset=0.2, valid=True)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "route.json")
            memory.save(path)
            loaded = RouteMemory.load(path)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded.points[0].offset, 0.2)

    def test_event_round_trip(self):
        memory = RouteMemory()
        memory.mark_event("road_lost", self.pose, 1.0)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "route.json")
            memory.save(path)
            loaded = RouteMemory.load(path)
        self.assertEqual(loaded.events[0]["kind"], "road_lost")

    def make_reference(self, curved=False, hazard=False):
        ref = RouteMemory(spacing_m=0.01)
        for i in range(11):
            p = DeadReckoner()
            p.s = i * 0.1
            p.x = i * 0.1
            p.y = 0.0
            p.theta = (i * 0.05 if curved else 0.0)
            ref.record(p, i * 0.1, valid=not (hazard and i >= 5))
        return ref

    def test_advisor_straight_route(self):
        advisor = RouteAdvisor(self.make_reference())
        advice = advisor.advice(0.0, 0.0, 0.0)
        self.assertTrue(advice.confident)
        self.assertFalse(advice.hazard_ahead)
        self.assertAlmostEqual(advice.speed_scale, 1.0)

    def test_advisor_curve_slowdown(self):
        advisor = RouteAdvisor(self.make_reference(curved=True))
        advice = advisor.advice(0.0, 0.0, 0.0)
        self.assertTrue(advice.confident)
        self.assertGreater(advice.curvature_ahead, 0.0)
        self.assertLessEqual(advice.speed_scale, 1.0)

    def test_advisor_road_loss_warning(self):
        advisor = RouteAdvisor(self.make_reference(hazard=True),
                               lookahead_m=0.8)
        advice = advisor.advice(0.4, 0.0, 0.4)
        self.assertTrue(advice.confident)
        self.assertTrue(advice.hazard_ahead)
        self.assertLessEqual(advice.speed_scale, 0.5)

    def test_advisor_empty_route(self):
        self.assertEqual(RouteAdvisor(RouteMemory()).advice(0, 0, 0),
                         NO_ADVICE)

    def test_bad_route_version_rejected(self):
        with self.assertRaises(ValueError):
            RouteMemory.from_dict({"version": 999, "points": []})

    def test_nonfinite_json_rejected(self):
        memory = RouteMemory()
        with self.assertRaises(ValueError):
            memory.mark_event("bad", self.pose, 0.0, bad=float("nan"))


if __name__ == "__main__":
    unittest.main()

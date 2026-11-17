"""Pure-Python route recording, dead reckoning, and replay advice.

This module deliberately has no ROS dependency so it can be unit-tested on a PC.
Replay advice is advisory only: it estimates upcoming curvature and remembered
road-loss regions. It does not generate steering or motor commands.
"""
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import NamedTuple

FORMAT_VERSION = 1


def _finite(*values):
    try:
        return all(v is not None and math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


class DeadReckoner:
    """Integrate gyro heading and signed forward wheel travel in a 2-D plane."""

    def __init__(self, max_gyro_dt=0.2):
        if not _finite(max_gyro_dt) or max_gyro_dt <= 0:
            raise ValueError("max_gyro_dt must be finite and positive")
        self.max_gyro_dt = float(max_gyro_dt)
        self.reset()

    def reset(self):
        self.x = self.y = self.theta = self.s = 0.0
        self.omega = 0.0
        self._gyro_t = None

    def update_gyro(self, omega_z, t):
        if not _finite(omega_z, t):
            return
        omega_z, t = float(omega_z), float(t)
        if self._gyro_t is not None:
            dt = t - self._gyro_t
            if 0 < dt <= self.max_gyro_dt:
                self.theta += 0.5 * (self.omega + omega_z) * dt
        self._gyro_t, self.omega = t, omega_z

    def update_distance(self, forward_m):
        if not _finite(forward_m):
            return
        forward_m = float(forward_m)
        self.x += forward_m * math.cos(self.theta)
        self.y += forward_m * math.sin(self.theta)
        self.s += abs(forward_m)


@dataclass
class RoutePoint:
    s: float
    x: float
    y: float
    theta: float
    t: float
    offset: float = None
    valid: bool = False
    yellow_px: int = None
    blue_left: float = None
    blue_right: float = None
    v: float = None
    omega: float = None


class RouteMemory:
    """Store route breadcrumbs and a coarse visited-cell map."""

    def __init__(self, spacing_m=0.05, cell_m=0.10):
        if not _finite(spacing_m, cell_m) or spacing_m <= 0 or cell_m <= 0:
            raise ValueError("spacing_m and cell_m must be finite and positive")
        self.spacing_m = float(spacing_m)
        self.cell_m = float(cell_m)
        self.points = []
        self.events = []
        self.visited = set()
        self.meta = {}

    def __len__(self):
        return len(self.points)

    def _cell(self, x, y):
        return (math.floor(x / self.cell_m), math.floor(y / self.cell_m))

    def record(self, pose, t, offset=None, valid=False, yellow_px=None,
               blue_left=None, blue_right=None, v=None, omega=None):
        required = (pose.s, pose.x, pose.y, pose.theta, t)
        if not _finite(*required):
            return False
        s, x, y, theta, t = map(float, required)
        if self.points:
            previous_s = self.points[-1].s
            if s < previous_s or s - previous_s < self.spacing_m:
                return False
        optional = (offset, blue_left, blue_right, v, omega)
        optional = [float(vv) if _finite(vv) else None for vv in optional]
        if yellow_px is not None:
            try:
                yellow_px = int(yellow_px)
            except (TypeError, ValueError):
                yellow_px = None
        point = RoutePoint(
            s=s, x=x, y=y, theta=theta, t=t,
            offset=optional[0], valid=bool(valid), yellow_px=yellow_px,
            blue_left=optional[1], blue_right=optional[2],
            v=optional[3], omega=optional[4],
        )
        self.points.append(point)
        self.visited.add(self._cell(x, y))
        return True

    def mark_event(self, kind, pose, t, **info):
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("event kind must be a non-empty string")
        if not _finite(pose.s, pose.x, pose.y, pose.theta, t):
            return False
        event = {
            "kind": kind.strip(),
            "s": float(pose.s),
            "x": float(pose.x),
            "y": float(pose.y),
            "theta": float(pose.theta),
            "t": float(t),
        }
        # Keep event metadata JSON-safe; serialization errors are caught here.
        event.update(info)
        json.dumps(event, allow_nan=False)
        self.events.append(event)
        return True

    def was_here(self, x, y, radius_cells=0):
        if not _finite(x, y) or radius_cells < 0:
            return False
        cx, cy = self._cell(float(x), float(y))
        r = int(radius_cells)
        return any((cx + dx, cy + dy) in self.visited
                   for dx in range(-r, r + 1)
                   for dy in range(-r, r + 1))

    def to_dict(self):
        return {
            "version": FORMAT_VERSION,
            "spacing_m": self.spacing_m,
            "cell_m": self.cell_m,
            "meta": self.meta,
            "events": self.events,
            "points": [asdict(p) for p in self.points],
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported or invalid route file version")
        memory = cls(data.get("spacing_m", 0.05), data.get("cell_m", 0.10))
        points = data.get("points", [])
        events = data.get("events", [])
        if not isinstance(points, list) or not isinstance(events, list):
            raise ValueError("Route points and events must be lists")
        for raw in points:
            if not isinstance(raw, dict):
                raise ValueError("Invalid route point")
            point = RoutePoint(**raw)
            if not _finite(point.s, point.x, point.y, point.theta, point.t):
                raise ValueError("Route point contains non-finite required values")
            if memory.points and point.s < memory.points[-1].s:
                raise ValueError("Route point distances must be nondecreasing")
            memory.points.append(point)
            memory.visited.add(memory._cell(point.x, point.y))
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("kind"), str):
                raise ValueError("Invalid route event")
            memory.events.append(event)
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError("Route metadata must be an object")
        memory.meta = meta
        return memory

    def save(self, path):
        path = os.path.expanduser(os.fspath(path))
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".route-", suffix=".tmp",
                                         dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path):
        with open(os.path.expanduser(os.fspath(path)), encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


class Advice(NamedTuple):
    speed_scale: float
    curvature_ahead: float
    hazard_ahead: bool
    matched_s: float
    match_dist: float
    confident: bool


NO_ADVICE = Advice(1.0, 0.0, False, 0.0, math.inf, False)


class RouteAdvisor:
    """Match current odometry to a reference route and estimate upcoming risk."""

    def __init__(self, reference, lookahead_m=0.5, window_m=0.4,
                 max_match_m=0.3, curve_slowdown=0.5, min_scale=0.4,
                 hazard_scale=0.5, straight_boost=1.0,
                 straight_kappa=0.2, smoothing_points=2):
        values = (lookahead_m, window_m, max_match_m, curve_slowdown,
                  min_scale, hazard_scale, straight_boost, straight_kappa)
        if not _finite(*values) or min(lookahead_m, window_m, max_match_m) <= 0:
            raise ValueError("RouteAdvisor distances and tuning values must be finite; distances positive")
        if not 0 < min_scale <= 1 or not 0 < hazard_scale <= 1:
            raise ValueError("speed scales must be in (0, 1]")
        self.ref = reference.points
        self.events = reference.events
        self.lookahead_m = float(lookahead_m)
        self.window_m = float(window_m)
        self.max_match_m = float(max_match_m)
        self.curve_slowdown = float(curve_slowdown)
        self.min_scale = float(min_scale)
        self.hazard_scale = float(hazard_scale)
        self.straight_boost = float(straight_boost)
        self.straight_kappa = float(straight_kappa)
        self.kappa = self._curvatures(max(1, int(smoothing_points)))
        self.reset()

    def reset(self):
        self._ref_s = None
        self._run_s = None

    def _curvatures(self, k):
        result = []
        n = len(self.ref)
        for i, point in enumerate(self.ref):
            a = self.ref[max(0, i - k)]
            b = self.ref[min(n - 1, i + k)]
            ds = b.s - a.s
            result.append((b.theta - a.theta) / ds if ds > 1e-6 else 0.0)
        return result

    def advice(self, x, y, run_s):
        if len(self.ref) < 3 or not _finite(x, y, run_s):
            return NO_ADVICE
        run_s = float(run_s)
        expected = run_s if self._ref_s is None else (
            self._ref_s + max(0.0, run_s - self._run_s)
        )
        self._run_s = run_s

        candidates = [
            (i, p) for i, p in enumerate(self.ref)
            if abs(p.s - expected) <= self.window_m
        ]
        if not candidates:
            self._ref_s = expected
            return NO_ADVICE._replace(matched_s=expected)
        best_i, best = min(candidates,
                           key=lambda item: math.hypot(item[1].x - x,
                                                       item[1].y - y))
        distance = math.hypot(best.x - x, best.y - y)
        if distance > self.max_match_m:
            self._ref_s = expected
            return NO_ADVICE._replace(matched_s=expected,
                                      match_dist=distance)

        self._ref_s = best.s
        lo, hi = best.s, best.s + self.lookahead_m
        indices = [i for i, p in enumerate(self.ref) if lo <= p.s <= hi]
        curvature = max((abs(self.kappa[i]) for i in indices), default=0.0)
        hazard = any(not self.ref[i].valid for i in indices)
        hazard = hazard or any(
            e.get("kind") == "road_lost" and lo <= float(e.get("s", -math.inf)) <= hi
            for e in self.events
        )
        scale = max(self.min_scale,
                    1.0 / (1.0 + self.curve_slowdown * curvature))
        if hazard:
            scale = min(scale, self.hazard_scale)
        elif curvature < self.straight_kappa:
            scale = self.straight_boost
        return Advice(scale, curvature, hazard, best.s, distance, True)

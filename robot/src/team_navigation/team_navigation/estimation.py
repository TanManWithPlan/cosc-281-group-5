"""Road estimator for team The Perceptrons (COSC 81/281, Fall 2026).

Owner: Michael Dong (estimation). Written with help from Claude; see the
team AI disclosure.

WHAT THIS DOES
    Perception looks at ONE camera frame and says where the road is. That
    number is noisy: glare, a missed frame, a speck of yellow on the floor.
    This module sits between perception and control and turns that noisy
    stream into a steadier best guess, plus a yes/no "is it safe to steer on
    this right now?" flag.

    perception (one frame) --> RoadEstimator.update(...) --> RoadEstimate --> planning/control

WHY IT'S A SEPARATE FILE
    It has no ROS imports and doesn't depend on any teammate's code. It takes
    plain numbers in and gives a small dataclass out. That means:
      - it can be tested on any laptop with plain Python (estimation_tools/),
      - it plugs into navigation.py with ~3 lines in update_estimate, no matter
        how perception's output ends up looking.

UNITS
    The offset is in perception's units. With Tanner's perception.py that is
    PerceptionResult.error: -1 (road far left) .. +1 (road far right), so
    0 = road centered and POSITIVE = ROAD TO THE RIGHT. max_jump uses the same
    units. If perception switches to pixels, scale max_jump to match (e.g.
    0.3 of a 640 px half-width = 192 px).

HOW TO CALL IT (once per control tick, e.g. 20 times a second)
    est = RoadEstimator()
    e = est.update(offset, found, image_stamp_s, now_s)
        offset         perception's number for this frame (ignored if found is False,
                       so it's fine that perception repeats its last value then)
        found          True if perception saw enough road in this frame
        image_stamp_s  when the camera CAPTURED this frame (seconds), or None if no
                       image has arrived yet
        now_s          current time (seconds), same clock as image_stamp_s
    e.offset  -> smoothed offset (None until the road has been seen once)
    e.valid   -> True if it's OK to steer on e.offset right now
    e.status  -> short word saying what happened this tick (handy in logs)
"""
import math
from dataclasses import dataclass


@dataclass
class RoadEstimate:
    """The estimator's answer for one tick.

    offset       Smoothed road offset (perception's units, + = road right).
                 Held at its last value while the road is missing.
                 None until the road has been seen at least once.
    valid        True if planning/control may steer on `offset` right now:
                 the newest image is fresh AND the road was seen recently.
    status       What happened this tick (see STATUSES below).
    raw          Perception's value this tick if found, else None (for logging).
    image_age_s  How old the newest camera frame is (inf if no image yet).
    road_age_s   How long since the road was last actually seen (inf if never).
    """
    offset: float | None
    valid: bool
    status: str
    raw: float | None
    image_age_s: float
    road_age_s: float


# What each status word means (they show up in logs and CSVs):
STATUSES = {
    'no_image':         'no camera frame has arrived yet',
    'stale_image':      'newest frame is too old to trust; estimate not updated',
    'repeat_frame':     'same frame as last tick; not counted twice',
    'first':            'first time seeing the road (or seeing it again after it was lost): start from this reading',
    'smoothed':         'normal case: blended the new reading in',
    'outlier_ignored':  'reading jumped too far from the estimate; ignored as a probable glitch',
    'outlier_accepted': 'the big jump repeated several frames in a row, so it is real: jump to it',
    'holding':          'road not seen this frame; keeping the last estimate for now',
    'lost':             'road missing longer than road_lost_timeout_s; estimate no longer valid',
}


class RoadEstimator:
    def __init__(self,
                 smoothing_alpha=0.2,
                 max_image_age_s=0.25,
                 road_lost_timeout_s=0.5,
                 max_jump=0.3,
                 outlier_accept_count=3):
        """All tunable numbers, with defaults so it works out of the box.

        smoothing_alpha      Weight on the NEW reading. 0.2 = "80% old + 20% new".
                             Bigger reacts faster but passes more noise through;
                             smaller is smoother but lags behind real turns.
        max_image_age_s      Frames older than this are not used (camera stalled
                             or lagging). The course safety node does NOT check
                             image age; only request age. This is our job.
        road_lost_timeout_s  How long to keep steering on the last estimate when
                             the road disappears, before declaring it lost.
                             0.5 s is about 15 frames at 30 fps.
        max_jump             A reading farther than this from the current estimate
                             (in one frame) is treated as a glitch...
        outlier_accept_count ...unless it happens this many frames IN A ROW, which
                             means the road really did move (e.g. a sharp turn).
        """
        self.smoothing_alpha = smoothing_alpha
        self.max_image_age_s = max_image_age_s
        self.road_lost_timeout_s = road_lost_timeout_s
        self.max_jump = max_jump
        self.outlier_accept_count = outlier_accept_count
        self.reset()

    def reset(self):
        """Forget everything (e.g. at the start of a new run)."""
        self.offset = None                # current best guess; None = never seen
        self.last_road_stamp_s = None     # capture time of the last frame WITH road
        self.last_used_stamp_s = None     # capture time of the last frame we looked at
        self.outlier_streak = 0           # how many big jumps in a row so far

    # ------------------------------------------------------------------
    def update(self, offset, found, image_stamp_s, now_s):
        """Fold in one tick's perception result and return a RoadEstimate."""
        raw = offset if found else None

        # --- 1. No image yet: nothing to do. ---
        if image_stamp_s is None:
            return self._answer('no_image', raw, math.inf, now_s)

        image_age_s = now_s - image_stamp_s

        # --- 2. Too old: don't let an old picture move the estimate. ---
        # If the camera freezes, the node keeps handing us the same last frame.
        # Steering on it would be steering blind, so we skip it, and `valid`
        # below goes False because image_age_s is too big.
        if image_age_s > self.max_image_age_s:
            return self._answer('stale_image', raw, image_age_s, now_s)

        # --- 3. Same frame as last tick: don't count it twice. ---
        # The control timer (20 Hz) and the camera (~30 fps) aren't in sync,
        # and frames get dropped, so sometimes we're handed the same frame
        # again. Blending it in twice would give one reading double weight.
        if self.last_used_stamp_s is not None and image_stamp_s <= self.last_used_stamp_s:
            return self._answer('repeat_frame', raw, image_age_s, now_s)
        self.last_used_stamp_s = image_stamp_s

        # --- 4. Road not in this frame: hold the last estimate. ---
        # We do NOT blend anything in. The estimate stays put, and _answer()
        # decides whether it's still usable based on how long it's been.
        if not found or raw is None or not math.isfinite(raw):
            status = 'holding'
            if self._road_age(now_s) > self.road_lost_timeout_s:
                status = 'lost'
            return self._answer(status, None, image_age_s, now_s)

        # --- 5. Road seen. Was it lost before this? Then start fresh. ---
        # If we've never seen it, or it was gone longer than the timeout, the
        # old estimate describes where the road USED to be. Averaging with it
        # would drag the new estimate toward stale information.
        road_was_lost = (self.offset is None or
                         image_stamp_s - self.last_road_stamp_s > self.road_lost_timeout_s)
        if road_was_lost:
            self.offset = raw
            self.outlier_streak = 0
            self.last_road_stamp_s = image_stamp_s
            return self._answer('first', raw, image_age_s, now_s)

        # --- 6. Glitch check. ---
        # The road can't really teleport across the image in 1/30 s. A huge
        # one-frame jump is almost always glare or a yellow blob on the floor.
        # Example: estimate 0.05, reading 0.60 -> jump 0.55 > 0.3 -> ignored.
        # But if the jump REPEATS (outlier_accept_count frames in a row), it's
        # real (a sharp turn), so accept it by jumping straight to the reading.
        if abs(raw - self.offset) > self.max_jump:
            self.outlier_streak += 1
            if self.outlier_streak < self.outlier_accept_count:
                # Ignored frames don't refresh last_road_stamp_s on purpose:
                # we only trust road sightings we actually used.
                return self._answer('outlier_ignored', raw, image_age_s, now_s)
            self.offset = raw
            self.outlier_streak = 0
            self.last_road_stamp_s = image_stamp_s
            return self._answer('outlier_accepted', raw, image_age_s, now_s)

        # --- 7. Normal case: exponential moving average (EMA). ---
        #   new = (1 - alpha) * old + alpha * reading
        # With alpha 0.2: old 0.10, reading 0.20 -> 0.8*0.10 + 0.2*0.20 = 0.12.
        # One noisy frame only moves the estimate 20% of the way toward it.
        # Cost: lag. After a real step from 0 to 1, the estimate goes
        # 0.2, 0.36, 0.49, 0.59, 0.67 -> about 5 frames (~0.17 s) to get 2/3 there.
        self.outlier_streak = 0
        a = self.smoothing_alpha
        self.offset = (1.0 - a) * self.offset + a * raw
        self.last_road_stamp_s = image_stamp_s
        return self._answer('smoothed', raw, image_age_s, now_s)

    # ------------------------------------------------------------------
    def _road_age(self, now_s):
        """Seconds since the road was last seen. Measured from NOW (not from
        the last frame) so that if frames stop coming at all, this keeps growing."""
        if self.last_road_stamp_s is None:
            return math.inf
        return now_s - self.last_road_stamp_s

    def _answer(self, status, raw, image_age_s, now_s):
        road_age_s = self._road_age(now_s)
        valid = (self.offset is not None and
                 image_age_s <= self.max_image_age_s and
                 road_age_s <= self.road_lost_timeout_s)
        return RoadEstimate(offset=self.offset, valid=valid, status=status, raw=raw,
                            image_age_s=image_age_s, road_age_s=road_age_s)

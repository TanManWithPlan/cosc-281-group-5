# Robot 5 navigation tuning notes (Nov 17, 2026)

Floor-test findings and the settings that came out of them. Branch `alejo/nav-tuning`,
built on `jetson5-centerline-wip` (a snapshot of the team's uncommitted Jetson code).

## Measurements

| Item | Value | How it was measured | Status |
| --- | --- | --- | --- |
| Encoder calibration | left 1.3151e-4, right 1.2987e-4 m/tick | team's 1 m straight run; validated to +0.05% / -0.95% | installed in `/etc/a2r-robot/encoder_calibration.json` |
| Track width (old) | 0.2416 m | team's 90° ground turns | still in `~/robot-setup-measurements/geometry.json` |
| **Track width (measured)** | **0.205 m** (0.214-0.218 at slower spins) | `tools/spin_test.py`: spin in place both directions, measured encoder tread speeds vs IMU gyro | **use this**; pass `track_width_m:=0.205` |

With 0.2416 the robot rotated 21-25% more than commanded in **both** directions (visually
~170° instead of ~137° per 4 s spin). With 0.205 the error is 2-5%. An earlier "left/right
asymmetry" reading was an artifact of two nodes publishing `navigation/request` at once.

**Team decision pending:** update `geometry.json` (used by `start-base.sh`) to 0.205-0.21.

## What the floor tests showed

1. Raw camera frames often arrived > 0.25 s old -> `stale_image` stops (stutter, worse with speed).
   The compressed topic arrives ~0.02-0.05 s old.
2. Fixed steering gain only worked at one speed: fast runs under-turned. Gains that tracked
   curves: 0.8 rad/s at 0.10 m/s, 2.4 at 0.30, 3.6 at 0.45 -> **~8 per metre** of speed.
3. After the width fix, the robot still turned too early and stopped too early at the road end:
   the target band sat far ahead and needed 50% of band rows to contain yellow.
4. At 0.80 m/s a near target band reacted too late -> the band must move farther ahead with speed.

## Navigation changes (all ROS parameters, live-settable)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `steering_gain_per_m` | 8.0 | yaw = -gain x speed x offset (0 = legacy fixed `steering_gain_rad_s`) |
| `max_yaw_rate_rad_s` | 2.4 | raise with speed (start-tuned.sh uses 5 x speed, 2.4-5.0) |
| `lookahead_band_top/bottom` | 0.45 / 0.85 | target band at <= `lookahead_slow_speed_m_s` (0.30) |
| `lookahead_fast_band_top/bottom` | 0.15 / 0.55 | target band at >= `lookahead_fast_speed_m_s` (0.80); linear blend between |
| `lookahead_min_coverage` | 0.3 | fraction of band rows that must contain yellow (was 0.5) |
| `follow_while_holding` / `holding_speed_scale` | true / 0.6 | keep following the held estimate while the road is briefly missing |
| `use_compressed_image` | true | subscribe to `camera/image_raw/compressed` (startup only) |
| `smoothing_alpha` | 0.2 | 0.35 at >= 0.45 m/s reacts faster |
| `road_lost_timeout_s` | 0.5 | 0.3 at >= 0.45 m/s (less blind driving past a road end) |

`perception.py` defaults are unchanged (0.25-0.55 band, 50% coverage), so `track_test`
and existing perception tests behave as before.

## How to run

On the Jetson, from a built checkout of this branch (robot 5 used `~/a2r-nav-alejo`):

```bash
bash tools/start-tuned.sh 0.30     # or 0.45, 0.80 ...
```

Press **Y** while stopped and neutral for Medium (> 0.15 m/s) or High (> 0.60 m/s). Hold **RB**.
Only one node may publish `navigation/request`; check with
`ros2 topic info /jetson5/navigation/request` (Publisher count: 1).

## Results so far

| Speed | Profile | Result |
| --- | --- | --- |
| 0.05-0.10 | Low | follows straights; cut corners before width fix |
| 0.30 (fixed gain 2.4) | Medium | tracked curves |
| 0.45 (fixed gain 3.6) | Medium | tracked curves, slight overturn |
| 0.30, tuned branch | Medium | **no overturn**, turns at the right time |
| 0.80, near band | High | reacted too late -> speed-dependent band added (not yet floor-tested) |

## Hardware notes

- The Jetson power bank shut down three times during testing (twice under load at speed).
  Charge or swap it before long sessions; the RVR battery also died once.
- `ros2 bag` could not record `a2r_robot_interfaces` topics (unknown type); use
  `ros2 topic echo --csv` for encoders.

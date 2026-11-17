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

On the Jetson (SSH), once per checkout: build with the course Python environment.

```bash
cd ~/a2r-student && git fetch && git checkout alejo/nav-route-merge
source /opt/ros/jazzy/setup.bash && source ~/ros-venv/bin/activate
cd robot && python -m colcon build --symlink-install     # expect: 4 packages finished
```

Then, from the repository root, one command starts the base (motors enabled, track width from
`~/robot-setup-measurements/geometry.json`) and navigation, with speed-matched settings:

```bash
bash tools/start-tuned.sh 0.30                              # normal: camera following only
bash tools/start-tuned.sh 0.30 record ~/routes/lap1.json    # same driving, also saves the route
bash tools/start-tuned.sh 0.30 replay ~/routes/lap1.json    # driving + route advice (slows before
                                                            # recorded curves / road-lost spots)
```

1. RVR powered on; F710 on **X**, MODE light off; sticks centered.
2. Press **Y** while stopped and neutral: Medium for > 0.15 m/s, High for > 0.60 m/s.
3. Put the robot on the tape. For record/replay, use the **same taped start mark facing the same
   way** and start the script with the robot already there: position is counted from where the
   node starts, and carrying/rotating the robot afterwards confuses replay.
4. Hold **RB** to drive; release (or press A) to stop. Ctrl+C in the script stops everything; in
   record mode it also does the final route save. Record one lap per file.

Live changes (another SSH terminal, after sourcing ROS + this workspace + `/etc/a2r-robot/robot.sh`):

```bash
ros2 param set /$ROBOT_NAMESPACE/navigation requested_forward_speed_m_s 0.45
ros2 param set /$ROBOT_NAMESPACE/navigation route_advice_weight 0.5     # replay: 0 = log only
ros2 param set /$ROBOT_NAMESPACE/navigation lookahead_band_top 0.55     # turn later
```

Only one node may publish `navigation/request`; check with
`ros2 topic info /$ROBOT_NAMESPACE/navigation/request` (Publisher count: 1). The script refuses to
start if a base or navigation node is already running. Logs: `~/nav_runs/run_*/nav_log.csv`
(includes `route_factor` and `route_s`).

## Results so far

| Speed | Profile | Result |
| --- | --- | --- |
| 0.05-0.10 | Low | follows straights; cut corners before width fix |
| 0.30 (fixed gain 2.4) | Medium | tracked curves |
| 0.45 (fixed gain 3.6) | Medium | tracked curves, slight overturn |
| 0.30, tuned branch | Medium | **no overturn**, turns at the right time |
| 0.80, near band | High | reacted too late -> speed-dependent band added |
| 0.30 record + replay (merged) | Medium | drove several laps fine; 179-point route, returned within 4 cm of start |

## Hardware notes

- The Jetson power bank shut down three times during testing (twice under load at speed).
  Charge or swap it before long sessions; the RVR battery also died once.
- `ros2 bag` could not record `a2r_robot_interfaces` topics (unknown type); use
  `ros2 topic echo --csv` for encoders.

## Route memory merge (Guillermo's `GR-Mapping&Foundation`)

Branch `alejo/nav-route-merge` = `alejo/nav-tuning` + Guillermo's route memory. His
`route_memory.py` (DeadReckoner, RouteMemory, RouteAdvisor) and its tests are unchanged; the
tuned navigation node now feeds it (gyro heading, encoder travel, estimator validity) instead of
his zero-motion node. His node's behaviour that was ported: overwrite protection, >= 3-point
replay check, hardware-session reset, 1 s periodic save plus final save on shutdown, half-size
JPEG decode.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `route_mode` | `off` | `record` = save breadcrumbs + road_lost/found events while driving; `replay` = load a route and use its advice (startup only) |
| `route_file` | `~/.ros/team_navigation/routes/test_route.json` | route JSON (startup only) |
| `route_advice_weight` | 1.0 | live: speed x (1 - weight x (1 - advice)); 0 = log advice only |

Advice only ever slows the robot (before known curves and where the road was lost when
recording); steering is still camera-driven. Route errors disable the route features, never driving.
CSV logs gain `route_factor` and `route_s` (distance travelled) columns.

```bash
# record a lap, then replay it with advice
ros2 launch team_navigation team.launch.py namespace:=jetson5   # after editing route_mode in the yaml, or:
ros2 run team_navigation navigation --ros-args -r __ns:=/jetson5 \
  --params-file robot/install/team_navigation/share/team_navigation/config/navigation.yaml \
  -p route_mode:=record -p route_file:=$HOME/routes/lap1.json
# ... drive the lap, Ctrl+C (final save), then -p route_mode:=replay
```

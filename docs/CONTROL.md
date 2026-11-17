# Road-following control

The control branch integrates `perception.py` from `origin/pixel-clustering`
(ab550a9) and `estimation.py` from `origin/estimation` (87a4853), unchanged.
The new controller and navigation adapter connect these existing team modules.
This does not include the perception branch's separate web debugging node.

Pipeline: BGR camera frame -> yellow/blue perception -> smoothed road offset ->
proportional controller -> `navigation/request` at 20 Hz. IMU is received but
not used. Offset is normalized image position, not metres or heading.

Positive offset means road right; positive ROS yaw means left, so
`yaw = clamp(-steering_kp * offset, +/- max_yaw_rate_rad_s)`.
Forward speed is `requested_forward_speed_m_s * (1 - turn_slowdown * abs(offset))`.
Defaults are an initial tuning proposal, not physically validated gains.

The controller immediately requests zero on missing yellow detection, blue
boundary warning, invalid estimate, future image timestamp, or image older than
`max_image_age_s` (0.25 s). Estimator holding never overrides missing detection.
There is no search/recovery motion or intersection planner. Tracking resumes
when fresh usable detections resume; the course base independently handles RB
arming, request expiry, and motor permission. No baseline safety code is changed.

Tune ROS parameters in `robot/src/team_navigation/config/navigation.yaml`.
HSV ranges must match the actual lighting and tape. Blue warning stops on either
side, so verify that normal boundary visibility does not cause constant stopping.
Do not infer physical distance from the image offset.

## Verification and tuning

From the repository root, install `requirements-test.txt` in a virtual environment
and run `python -m pytest tests -q`. Tests cover signs, limits, slowdown, bad inputs,
synthetic-image integration, and mocked ROS request publication.

On the Jetson rebuild the robot workspace, following docs/SETUP.md. Launch the
base first with `motion_enabled:=false` and the measured `track_width_m`, then
launch team_navigation. Inspect `navigation/request`: centered tape should request
straight travel, tape right negative yaw, tape left positive yaw. Cover the camera,
remove tape, and show blue tape to verify zero requests. Verify stale-camera stops.

After the course manual stopping/calibration checks, validate with supervised Low
speed and the documented RB release procedure. Start at 0.05 m/s, adjust steering
P gain gradually, and log offset, speed, yaw and tracking failures. Add derivative
control only if measured oscillation warrants it. Hardware/ROS transport and actual
road-following performance require robot testing; laptop tests cannot validate them.

## AI disclosure

Codex assisted with control implementation, navigation integration, documentation,
and hardware-free tests. Review and validate the generated changes before robot use.
The imported estimation module retains its author's AI attribution.

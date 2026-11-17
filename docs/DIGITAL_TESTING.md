# Digital integration and driving tests

No paper track, robot, or ROS installation is required for these checks.

## Integrated branches

`integration/digital-tests` combines control, drive-basic (including estimation
and latest pixel-clustering), GR-Mapping&Foundation, and testing-changes-bella.
Overlapping navigation implementations were resolved in favor of drive-basic's
fork selection and CSV logging with control.py for motion. The mapping branch's
adapter is superseded; it contained no independent mapping implementation.
Bella's unfinished navigation tests were replaced with executable behavioral tests;
the supplied image fixtures remain. No motor/safety baseline is changed.

## Run on a laptop

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python tools/digital_drive.py --output simulation-results
```

The simulator writes one CSV per scenario and summary.json. Keep generated
results out of git. It runs 80 simulated seconds at 20 Hz for each of seven cases:
initial left/right displacement, gentle S curve, noisy/dim curve, a frozen camera,
missing tape, and a blue boundary. The latter three occur at t=10..11 seconds.
The same seeded images give repeatable results. Assertions check final error below
4 cm, maximum error below 16 cm, and no nonzero requests when missing detection,
blue warnings, or images older than 0.25 s require stopping. Recovery is checked.

## What is actually exercised

1. Unit tests: steering signs, limits, turn slowdown, invalid inputs/parameters.
2. Mocked ROS adapter: real image processing, estimation, fork selection, request
   publication, frozen/future/out-of-order images, clock rollback, and CSV output.
3. Closed loop: real perception, estimator, and controller with synthetic camera
   frames and ideal planar differential-drive kinematics. ROS transport and the
   navigation adapter are tested separately, not run inside this simulator.

## Model limits and next digital experiments

Synthetic tape is rendered using a simplified camera projection onto flat ground.
The robot follows velocity requests perfectly. There is no tire slip, encoder
noise, latency model, 3D scene, hardware gate, or actuator model. Simulated errors
are model results, not measured physical accuracy. They show consistency and catch
regressions; they do not establish real-world driving readiness.

Next use actual camera recordings (a screen-displayed scene can help check the
pipeline without paper, but is not a geometric driving test). Check HSV masks,
field of view, compression, and exposure against those recordings. Add latency,
actuator response and camera geometry only when their measured values are known.
Intersections are only tested for blob selection; no global route planner exists.

HSV tuning uses hsv_config_path JSON, shared with track_test's saved settings.
Parameters are refreshed each tick, except HSV file loading and CSV startup settings,
which require restart. stop_on_blue defaults true; switching it off enables
blue_slowdown_factor instead and must be deliberate. Invalid live control settings
cause zero requests. The default forward speed remains 0.05 m/s.

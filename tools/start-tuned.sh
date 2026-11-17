#!/usr/bin/env bash
# Start the robot base + tuned navigation (with optional route record/replay).
#
#   bash tools/start-tuned.sh [SPEED_M_S] [off|record|replay] [ROUTE_FILE]
#
#   bash tools/start-tuned.sh 0.30                               # normal driving, no route
#   bash tools/start-tuned.sh 0.30 record ~/routes/lap1.json     # drive and save a route
#   bash tools/start-tuned.sh 0.30 replay ~/routes/lap1.json     # drive with route advice
#
# Run on the Jetson from a checkout whose robot/ workspace is already built.
# Track width comes from ~/robot-setup-measurements/geometry.json (TRACK_WIDTH_M overrides).
# Press Y while stopped and neutral: Medium for > 0.15 m/s, High for > 0.60 m/s. Hold RB to drive.
# Ctrl+C stops everything (and does the final route save in record mode).
set -e
SPEED=${1:-0.30}
MODE=${2:-off}
ROUTE_FILE=${3:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GEOMETRY=~/robot-setup-measurements/geometry.json

case "$MODE" in off|record|replay) ;; *) echo "Mode must be off, record or replay"; exit 1;; esac
if [ "$MODE" != off ] && [ -z "$ROUTE_FILE" ]; then
  echo "Give a route file for $MODE, e.g. ~/routes/lap1.json"; exit 1; fi
if [ "$MODE" = replay ] && [ ! -f "$ROUTE_FILE" ]; then
  echo "Route file not found: $ROUTE_FILE (record one first)"; exit 1; fi
if [ "$MODE" = record ] && [ -e "$ROUTE_FILE" ]; then
  echo "Route file exists: $ROUTE_FILE (pick a new name so it is not overwritten)"; exit 1; fi

if [ -n "$TRACK_WIDTH_M" ]; then
  WIDTH=$TRACK_WIDTH_M
else
  WIDTH=$(python3 -c "import json,math,os; v=json.load(open(os.path.expanduser('$GEOMETRY')))['track_width_m']; assert math.isfinite(v) and v>0; print(v)")
fi

source /opt/ros/jazzy/setup.bash
source "$ROOT/robot/install/setup.bash"
source ~/ros-venv/bin/activate
source /etc/a2r-robot/robot.sh

if pgrep -f '[/]opt/ros/jazzy/bin/ros2 launch a2r_robot robot.launch.py' >/dev/null; then
  echo 'A base launch is already running. Stop it first.'; exit 1; fi
if pgrep -f '[t]eam_navigation/lib/team_navigation/navigation' >/dev/null; then
  echo 'A navigation node is already running. Stop it first (only one may publish navigation/request).'; exit 1; fi

systemctl --user stop rvr-camera.service || true
ros2 launch a2r_robot robot.launch.py namespace:="$ROBOT_NAMESPACE" motion_enabled:=true track_width_m:=$WIDTH &
BASE=$!
sleep 8

ROUTE_ARGS=()
if [ "$MODE" != off ]; then
  mkdir -p "$(dirname "$ROUTE_FILE")"
  ROUTE_ARGS=(-p route_mode:=$MODE -p route_file:=$ROUTE_FILE)
fi
ros2 run team_navigation navigation --ros-args -r __ns:=/$ROBOT_NAMESPACE \
  --params-file "$ROOT/robot/install/team_navigation/share/team_navigation/config/navigation.yaml" \
  "${ROUTE_ARGS[@]}" &
NAV=$!
trap 'kill -INT $NAV 2>/dev/null; wait $NAV 2>/dev/null; kill -INT $BASE 2>/dev/null; wait' INT TERM
sleep 6

# Speed-matched settings: higher speed -> higher turn cap, shorter blind hold, faster smoothing.
read -r YAW HOLD ALPHA < <(python3 -c "
v=float('$SPEED')
print(max(2.4, min(5.0, 5.0*v)), 0.5 if v<=0.45 else 0.3, 0.2 if v<=0.45 else 0.35)")
for p in "max_yaw_rate_rad_s $YAW" "road_lost_timeout_s $HOLD" "smoothing_alpha $ALPHA" \
         "requested_forward_speed_m_s $SPEED"; do
  ros2 param set "/$ROBOT_NAMESPACE/navigation" $p >/dev/null && echo "set $p"
done
echo "Ready: speed $SPEED m/s, track width $WIDTH m, route mode $MODE ${ROUTE_FILE}."
echo "Hold RB to drive; Ctrl+C here stops everything."
wait

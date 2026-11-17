#!/usr/bin/env bash
# Start robot 5 with the tuned navigation (branch alejo/nav-tuning).
#   bash tools/start-tuned.sh [SPEED_M_S]      default 0.30
# Run on the Jetson from a checkout whose robot/ workspace is already built.
# Starts the base (motors enabled, track_width_m 0.205 from the spin test) and the
# tuned navigation from this checkout, then applies speed-matched settings.
# Ctrl+C stops both. Press Y for Medium (>0.15 m/s) or High (>0.60 m/s) while stopped.
set -e
SPEED=${1:-0.30}
WIDTH=${TRACK_WIDTH_M:-0.205}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/jazzy/setup.bash
source "$ROOT/robot/install/setup.bash"
source ~/ros-venv/bin/activate
source /etc/a2r-robot/robot.sh

if pgrep -f '[/]opt/ros/jazzy/bin/ros2 launch a2r_robot robot.launch.py' >/dev/null; then
  echo 'A base launch is already running. Stop it first.'; exit 1; fi
if pgrep -f '[r]os2 launch team_navigation team.launch.py' >/dev/null; then
  echo 'A navigation launch is already running. Stop it first.'; exit 1; fi

systemctl --user stop rvr-camera.service || true
ros2 launch a2r_robot robot.launch.py namespace:="$ROBOT_NAMESPACE" motion_enabled:=true track_width_m:=$WIDTH &
BASE=$!
sleep 8
ros2 launch team_navigation team.launch.py namespace:="$ROBOT_NAMESPACE" &
NAV=$!
trap 'kill -INT $NAV $BASE 2>/dev/null; wait' INT TERM
sleep 6

# Speed-matched settings: higher speed -> higher turn cap, shorter blind hold, faster smoothing.
read -r YAW HOLD ALPHA < <(python3 -c "
v=float('$SPEED')
print(max(2.4, min(5.0, 5.0*v)), 0.5 if v<=0.45 else 0.3, 0.2 if v<=0.45 else 0.35)")
for p in "max_yaw_rate_rad_s $YAW" "road_lost_timeout_s $HOLD" "smoothing_alpha $ALPHA" \
         "requested_forward_speed_m_s $SPEED"; do
  ros2 param set "/$ROBOT_NAMESPACE/navigation" $p >/dev/null && echo "set $p"
done
echo "Ready: speed $SPEED m/s, track width $WIDTH m. Hold RB to drive; Ctrl+C here stops everything."
wait

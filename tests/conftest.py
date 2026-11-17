"""Shared hardware-free ROS fixture with controllable clock and parameters."""
import pytest
import types
import importlib.util
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'robot/src/team_navigation'))
@pytest.fixture
def navigation(monkeypatch, tmp_path):
    """Exercise the actual ROS adapter with minimal in-process message mocks."""
    class Message:
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None)

    class Node:
        def __init__(self, name):
            self.subscriptions = {}
            self.messages = []
            self.parameters = {}
            self.now_ns = 1_000_000_000
        def declare_parameter(self, name, value):
            if name == 'csv_logging': value = False
            if name == 'hsv_config_path': value = str(tmp_path / 'hsv.json')
            self.parameters[name] = types.SimpleNamespace(value=value)
            return self.parameters[name]
        def get_parameter(self, name):
            return self.parameters[name]
        def create_subscription(self, kind, topic, callback, qos):
            self.subscriptions[topic] = callback
        def create_publisher(self, *args):
            return types.SimpleNamespace(publish=self.messages.append)
        def create_timer(self, period, callback):
            assert period == .05
        def get_clock(self):
            return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=self.now_ns, to_msg=lambda: 'stamp'))
        def get_logger(self):
            return types.SimpleNamespace(error=lambda text: None, info=lambda text: None, warning=lambda *a, **k: None)

    modules = {
        'rclpy': {}, 'rclpy.node': {'Node': Node},
        'rclpy.qos': {'qos_profile_sensor_data': 'sensor',
                     'QoSProfile': lambda **kw: kw,
                     'ReliabilityPolicy': types.SimpleNamespace(BEST_EFFORT='best_effort')},
        'rclpy.executors': {'ExternalShutdownException': type('Shutdown', (Exception,), {})},
        'sensor_msgs.msg': {'Image': Message, 'Imu': Message},
        'cv_bridge': {'CvBridge': lambda: types.SimpleNamespace(imgmsg_to_cv2=lambda msg, **kw: msg.data)},
        'course_robot_interfaces.msg': {'MotionRequest': Message},
    }
    for name, values in modules.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, mod)
    path = ROOT/'robot/src/team_navigation/team_navigation/navigation.py'
    spec = importlib.util.spec_from_file_location('project_navigation_test', path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod.Navigation()


"""The ROS 2 node.

Publishes   /affect/state      std_msgs/String  AffectState as JSON
            /affect/command    std_msgs/String  ExpressionCommand as JSON, limit-checked
Subscribes  /affect/appraisal  std_msgs/String  AppraisalEvent as JSON
            /affect/cue        std_msgs/String  a cue name, such as "nod"
Services    /affect/stop       std_srvs/Trigger the robot rests and commands are refused
            /affect/resume     std_srvs/Trigger

JSON on String keeps the versioned message schema in one place, with no colcon build.
The robot's own controller must still enforce its limits, and the hardware e-stop
bypasses all of this.
"""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from . import AffectBridge


class AffectNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("affect_engine")
        self._state_pub = self.create_publisher(String, "/affect/state", 10)
        self._command_pub = self.create_publisher(String, "/affect/command", 10)
        self.bridge = AffectBridge(
            lambda text: self._state_pub.publish(String(data=text)),
            lambda text: self._command_pub.publish(String(data=text)),
            personality=args.personality,
            profile=args.profile,
            tick_hz=args.tick_hz,
            log_dir=args.log_dir,
            warn=self.get_logger().warning,
        )
        self.create_subscription(String, "/affect/appraisal", self._on_appraisal, 10)
        self.create_subscription(String, "/affect/cue", self._on_cue, 10)
        self.create_service(Trigger, "/affect/stop", self._on_stop)
        self.create_service(Trigger, "/affect/resume", self._on_resume)
        self._start = self.get_clock().now()
        self.create_timer(1.0 / args.tick_hz, self._tick)
        self.get_logger().info(f"affect engine up: {args.personality} on {args.profile}")

    def _tick(self) -> None:
        elapsed = (self.get_clock().now() - self._start).nanoseconds / 1e9
        self.bridge.tick(elapsed)

    def _on_appraisal(self, message: String) -> None:
        self.bridge.on_appraisal(message.data)

    def _on_cue(self, message: String) -> None:
        if not self.bridge.on_cue(message.data):
            self.get_logger().warning(f"this robot has no cue {message.data!r}")

    def _on_stop(self, _request, response):
        self.bridge.stop()
        response.success = True
        response.message = "stopped; commands are refused until /affect/resume"
        return response

    def _on_resume(self, _request, response):
        self.bridge.resume()
        response.success = True
        response.message = "resumed"
        return response


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="affect-ros")
    parser.add_argument("--personality", default="pet")
    parser.add_argument("--profile", default="quadruped")
    parser.add_argument("--tick-hz", type=float, default=30.0)
    parser.add_argument("--log-dir")
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = AffectNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.bridge.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

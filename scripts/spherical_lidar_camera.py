#!/usr/bin/env python3
"""
Select 360 lidar points inside an angular box in camera_360_link.

The usual call flow is:
  1. read xyz points from the lidar message in the 360 lidar frame
  2. look up the TF transform from camera_360_link <- lidar_360
  3. transform points into camera_360_link
  4. keep points whose azimuth/elevation are inside the spherical box

ROS subscriber callback example:

    transform = lookup_panolidar_from_lidar_360_transform(
        self.tf_buffer,
        msg.header.stamp,
        lidar_360_frame=msg.header.frame_id or "lidar_360",
        panolidar_frame="camera_360_link",
    )
    selected_points, input_mask = lidar_points_in_spherical_bbox(
        msg,
        transform,
        azimuth=(-15.0, 15.0),
        elevation=(-10.0, 8.0),
    )
"""

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from builtin_interfaces.msg import Time as RosTime
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    import rclpy
    import tf2_ros
    from rosidl_runtime_py.utilities import get_message
except ImportError:  # Allows importing the math helpers outside a ROS environment.
    Duration = None
    Header = None
    HistoryPolicy = None
    Node = object
    PointCloud2 = None
    QoSProfile = None
    ReliabilityPolicy = None
    RosTime = None
    get_message = None
    point_cloud2 = None
    rclpy = None
    tf2_ros = None

AngleRange = Tuple[float, float]
WINDOW_NAME = "camera_360_link spherical point selector"
DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_OUTPUT_TOPIC = "/vis_selected_points"
cv2 = None


@dataclass(frozen=True)
class SphericalBoundingBox:
    """Angular/range limits in the target frame, normally camera_360_link."""

    azimuth: AngleRange
    elevation: AngleRange
    degrees: bool = True
    min_range: float = 0.0
    max_range: Optional[float] = None


def xyz_array_from_lidar_msg(msg) -> np.ndarray:
    """
    Return an Nx3 float array from either sensor_msgs/PointCloud2 or 360 lidar CustomMsg.

    The returned coordinates are still in msg.header.frame_id, usually the 360 lidar frame.
    """
    if PointCloud2 is not None and isinstance(msg, PointCloud2):
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        return np.asarray([(float(x), float(y), float(z)) for x, y, z in points], dtype=np.float64)

    # 360 lidar ROS 2 driver CustomMsg: msg.points contains x/y/z.
    if hasattr(msg, "points"):
        return np.asarray(
            [(float(point.x), float(point.y), float(point.z)) for point in msg.points],
            dtype=np.float64,
        )

    raise TypeError("Unsupported lidar message type: {}".format(type(msg).__name__))


def transform_points(points_xyz: np.ndarray, transform_stamped) -> np.ndarray:
    """
    Transform an Nx3 array using a geometry_msgs/TransformStamped.

    Pass the transform returned by:
      tf_buffer.lookup_transform("camera_360_link", "lidar_360", stamp)

    That transform means target <- source, so the output points are in camera_360_link.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)

    transform = transform_stamped.transform
    translation = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=np.float64,
    )
    rotation = quaternion_to_rotation_matrix(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )
    return points_xyz @ rotation.T + translation


def points_in_spherical_bbox(
    points_xyz_in_target: np.ndarray,
    bbox: SphericalBoundingBox,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select points inside an azimuth/elevation box.

    Angles are measured in the target frame:
      azimuth   = atan2(y, x)
      elevation = atan2(z, sqrt(x*x + y*y))

    Returns (selected_points, mask). Ranges may wrap around, so an azimuth range
    like (170, -170) means the 20 degree band across +/-180 degrees.
    """
    points_xyz_in_target = np.asarray(points_xyz_in_target, dtype=np.float64)
    if points_xyz_in_target.size == 0:
        return points_xyz_in_target.reshape(0, 3), np.zeros((0,), dtype=bool)

    x = points_xyz_in_target[:, 0]
    y = points_xyz_in_target[:, 1]
    z = points_xyz_in_target[:, 2]

    xy_range = np.hypot(x, y)
    point_range = np.linalg.norm(points_xyz_in_target, axis=1)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, xy_range)

    az_min, az_max = _angle_range_to_radians(bbox.azimuth, bbox.degrees)
    el_min, el_max = _angle_range_to_radians(bbox.elevation, bbox.degrees)

    mask = (
        _angle_in_range(azimuth, az_min, az_max)
        & _angle_in_range(elevation, el_min, el_max)
        & (point_range >= bbox.min_range)
    )
    if bbox.max_range is not None:
        mask &= point_range <= bbox.max_range

    return points_xyz_in_target[mask], mask


def lidar_points_in_spherical_bbox(
    msg,
    transform_target_from_lidar,
    azimuth: AngleRange,
    elevation: AngleRange,
    degrees: bool = True,
    min_range: float = 0.0,
    max_range: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for a lidar message published in the lidar frame.

    Returns selected points in camera_360_link plus a boolean mask aligned with the input
    xyz array produced from the message.
    """
    points_lidar = xyz_array_from_lidar_msg(msg)
    points_target = transform_points(points_lidar, transform_target_from_lidar)
    bbox = SphericalBoundingBox(
        azimuth=azimuth,
        elevation=elevation,
        degrees=degrees,
        min_range=min_range,
        max_range=max_range,
    )
    return points_in_spherical_bbox(points_target, bbox)


def lookup_target_from_lidar_transform(
    tf_buffer,
    stamp,
    lidar_frame: str = "livox_link",
    target_frame: str = "z1_link",
    timeout_sec: float = 0.05,
):
    """
    Look up the transform needed by lidar_points_in_spherical_bbox().

    In a ROS 2 node, create a tf2_ros.Buffer and call this with msg.header.stamp.
    """
    if Duration is None:
        return tf_buffer.lookup_transform(target_frame, lidar_frame, stamp)
    return tf_buffer.lookup_transform(target_frame, lidar_frame, stamp, timeout=Duration(seconds=timeout_sec))


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Cannot build a rotation matrix from a zero-length quaternion")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _angle_range_to_radians(angle_range: AngleRange, degrees: bool) -> AngleRange:
    low, high = angle_range
    if degrees:
        low = math.radians(low)
        high = math.radians(high)
    return _wrap_to_pi(low), _wrap_to_pi(high)


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _angle_in_range(angles: np.ndarray, low: float, high: float) -> np.ndarray:
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    if low <= high:
        return (wrapped >= low) & (wrapped <= high)
    return (wrapped >= low) | (wrapped <= high)


def filter_xyz_iterable_in_spherical_bbox(
    points_lidar: Iterable[Sequence[float]],
    transform_target_from_lidar,
    bbox: SphericalBoundingBox,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Variant for already-decoded points. Each input point may be xyz or xyz plus extras.
    """
    xyz = np.asarray([point[:3] for point in points_lidar], dtype=np.float64)
    points_target = transform_points(xyz, transform_target_from_lidar)
    return points_in_spherical_bbox(points_target, bbox)


class SphericalPointSelector(Node):
    """ROS 2 node with OpenCV sliders for live spherical point selection."""

    def __init__(self, args):
        super().__init__("panolidar_spherical_point_selector")
        self.args = args
        self.azimuth = (args.azimuth_min, args.azimuth_max)
        self.elevation = (args.elevation_min, args.elevation_max)
        self.last_input_count = 0
        self.last_selected_count = 0
        self.last_error = ""
        self.last_warn_time = None
        self.slider_ready = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=args.queue_depth,
        )

        lidar_msg_type = self.resolve_topic_type(args.lidar_topic)
        self.subscription = self.create_subscription(lidar_msg_type, args.lidar_topic, self.lidar_callback, qos)
        self.publisher = self.create_publisher(PointCloud2, args.output_topic, args.queue_depth)

        self.create_timer(0.05, self.update_slider_window)
        self.create_slider_window()

        self.get_logger().info("Subscribing to {}".format(args.lidar_topic))
        self.get_logger().info("Publishing selected {} points to {}".format(args.target_frame, args.output_topic))
        self.get_logger().info("Target frame: {}, source fallback: {}".format(args.target_frame, args.source_frame))

    def resolve_topic_type(self, topic_name):
        deadline = self.get_clock().now() + Duration(seconds=self.args.topic_wait_timeout)
        while rclpy.ok() and self.get_clock().now() < deadline:
            for name, topic_types in self.get_topic_names_and_types():
                if name == topic_name and topic_types:
                    return get_message(topic_types[0])
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("Could not determine topic type for {}".format(topic_name))

    def create_slider_window(self):
        global cv2
        if cv2 is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError("OpenCV is required for the slider window. Install python3-opencv or cv2.") from exc
            cv2 = cv2_module

        if cv2 is None:
            raise RuntimeError("OpenCV is required for the slider window. Install python3-opencv or cv2.")

        self.slider_ready = False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 760, 260)
        cv2.createTrackbar("azimuth min", WINDOW_NAME, self.deg_to_az_slider(self.azimuth[0]), 360, self.on_slider)
        cv2.createTrackbar("azimuth max", WINDOW_NAME, self.deg_to_az_slider(self.azimuth[1]), 360, self.on_slider)
        cv2.createTrackbar("elevation min", WINDOW_NAME, self.deg_to_el_slider(self.elevation[0]), 180, self.on_slider)
        cv2.createTrackbar("elevation max", WINDOW_NAME, self.deg_to_el_slider(self.elevation[1]), 180, self.on_slider)
        self.slider_ready = True
        self.on_slider(0)

    def on_slider(self, _value):
        if not self.slider_ready:
            return

        self.azimuth = (
            self.az_slider_to_deg(cv2.getTrackbarPos("azimuth min", WINDOW_NAME)),
            self.az_slider_to_deg(cv2.getTrackbarPos("azimuth max", WINDOW_NAME)),
        )
        self.elevation = (
            self.el_slider_to_deg(cv2.getTrackbarPos("elevation min", WINDOW_NAME)),
            self.el_slider_to_deg(cv2.getTrackbarPos("elevation max", WINDOW_NAME)),
        )

    def update_slider_window(self):
        canvas = np.full((260, 760, 3), 30, dtype=np.uint8)
        lines = [
            "Frame: {}    Topic: {}".format(self.args.target_frame, self.args.output_topic),
            "Azimuth:   {:+.1f} to {:+.1f} deg".format(*self.azimuth),
            "Elevation: {:+.1f} to {:+.1f} deg".format(*self.elevation),
            "Selected: {} / {} points".format(self.last_selected_count, self.last_input_count),
        ]
        if self.last_error:
            lines.append("Status: {}".format(self.last_error))
        else:
            lines.append("Status: receiving and publishing")

        for index, line in enumerate(lines):
            color = (80, 220, 255) if index in (1, 2) else (235, 235, 235)
            if index == 4 and self.last_error:
                color = (80, 120, 255)
            cv2.putText(
                canvas,
                line,
                (24, 44 + index * 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            rclpy.shutdown()

    def lidar_callback(self, msg):
        source_frame = getattr(getattr(msg, "header", None), "frame_id", "") or self.args.source_frame
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            stamp = RosTime()

        try:
            points_lidar = xyz_array_from_lidar_msg(msg)
            transform = lookup_target_from_lidar_transform(
                self.tf_buffer,
                stamp,
                lidar_frame=source_frame,
                target_frame=self.args.target_frame,
                timeout_sec=self.args.tf_timeout,
            )
            points_target = transform_points(points_lidar, transform)
            selected_points, _mask = points_in_spherical_bbox(
                points_target,
                SphericalBoundingBox(
                    azimuth=self.azimuth,
                    elevation=self.elevation,
                    min_range=self.args.min_range,
                    max_range=self.args.max_range,
                ),
            )
            self.publish_points(selected_points, stamp)
            self.last_input_count = len(points_lidar)
            self.last_selected_count = len(selected_points)
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
            now = self.get_clock().now()
            if self.last_warn_time is None or (now - self.last_warn_time).nanoseconds > 2_000_000_000:
                self.get_logger().warn(self.last_error)
                self.last_warn_time = now

    def publish_points(self, points_xyz, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.args.target_frame
        cloud = point_cloud2.create_cloud_xyz32(header, np.asarray(points_xyz, dtype=np.float32).tolist())
        self.publisher.publish(cloud)

    @staticmethod
    def az_slider_to_deg(value):
        return float(value - 180)

    @staticmethod
    def deg_to_az_slider(value):
        return int(round(max(-180.0, min(180.0, value)) + 180.0))

    @staticmethod
    def el_slider_to_deg(value):
        return float(value - 90)

    @staticmethod
    def deg_to_el_slider(value):
        return int(round(max(-90.0, min(90.0, value)) + 90.0))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter 360 lidar points by camera_360_link azimuth/elevation sliders and publish a PointCloud2."
    )
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--output-topic", default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--target-frame", default="z1_link")
    parser.add_argument("--source-frame", default="livox_link", help="Fallback source frame if the lidar message frame_id is empty.")
    parser.add_argument("--azimuth-min", type=float, default=-15.0)
    parser.add_argument("--azimuth-max", type=float, default=15.0)
    parser.add_argument("--elevation-min", type=float, default=-10.0)
    parser.add_argument("--elevation-max", type=float, default=10.0)
    parser.add_argument("--min-range", type=float, default=0.0)
    parser.add_argument("--max-range", type=float, default=None)
    parser.add_argument("--queue-depth", type=int, default=10)
    parser.add_argument("--tf-timeout", type=float, default=0.05)
    parser.add_argument("--topic-wait-timeout", type=float, default=10.0)
    return parser.parse_args()


def main():
    if rclpy is None or tf2_ros is None or PointCloud2 is None:
        raise RuntimeError("This node must be run in a ROS 2 Python environment.")

    args = parse_args()
    rclpy.init()
    node = SphericalPointSelector(args)
    try:
        rclpy.spin(node)
    finally:
        if cv2 is not None:
            cv2.destroyWindow(WINDOW_NAME)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

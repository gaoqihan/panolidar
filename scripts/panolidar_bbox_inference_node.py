#!/usr/bin/env python3
"""
ROS 2 bbox inference node for 360 camera image boxes and 360 lidar point clouds.

The node subscribes to the lidar topic, caches the latest cloud, and exposes
three services:
  /bbox_depth
  /bbox_position
  /bbox_bearing

Each service request supplies a pixel bbox in the 360 camera equirectangular image.
The node maps the bbox to the calibrated spherical camera model, selects lidar
points inside it, optionally filters outliers, and reduces the remaining points
using one of:
  center_k      average the k points closest to the bbox center ray
  all           average every selected point
  main_cluster  average the largest Euclidean cluster
"""

import argparse
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from spherical_lidar_camera import transform_points, xyz_array_from_lidar_msg

try:
    from panolidar.srv import PanoBBoxBearing, PanoBBoxDepth, PanoBBoxPosition
except ImportError as exc:
    raise RuntimeError(
        "Custom services are required. Add the srv/*.srv files to a ROS 2 "
        "Python package and build it with rosidl_default_generators."
    ) from exc


DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_SOURCE_FRAME = "livox_link"
DEFAULT_CAMERA_FRAME = "z1_link"
DEFAULT_OUTPUT_FRAME = "base_link"
DEFAULT_SELECTED_POINTS_TOPIC = "/vis_bbox_selected_points"
DEFAULT_USED_POINTS_TOPIC = "/vis_bbox_used_points"
DEFAULT_OUTLIER_POINTS_TOPIC = "/vis_bbox_outlier_points"
DEFAULT_RESULT_MARKER_TOPIC = "/vis_bbox_result_marker"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "config" / "panolidar_calibration_params.json"

METHOD_CENTER_K = "center_k"
METHOD_ALL = "all"
METHOD_MAIN_CLUSTER = "main_cluster"
OUTLIER_NONE = "none"
OUTLIER_MAD_RANGE = "mad_range"
OUTLIER_STATISTICAL = "statistical"
OUTLIER_RADIUS = "radius"


@dataclass
class InferenceResult:
    valid: bool
    message: str
    input_count: int
    selected_count: int
    used_count: int
    point: Optional[np.ndarray]
    stamp: object
    frame_id: str

    @property
    def depth_m(self) -> float:
        if self.point is None:
            return float("nan")
        return float(np.linalg.norm(self.point))

    @property
    def xy_distance_m(self) -> float:
        if self.point is None:
            return float("nan")
        return float(np.hypot(self.point[0], self.point[1]))

    @property
    def angle_rad(self) -> float:
        if self.point is None:
            return float("nan")
        return float(math.atan2(self.point[1], self.point[0]))


class BBoxInferenceNode(Node):
    def __init__(self, args):
        super().__init__("panolidar_bbox_inference")
        self.args = args
        self.lock = threading.Lock()
        self.latest_points_camera = None
        self.latest_stamp = None
        self.latest_input_count = 0
        self.rotation = load_calibration_rotation(args.calibration_path)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=args.queue_depth,
        )
        lidar_msg_type = self.resolve_topic_type(args.lidar_topic)
        self.create_subscription(lidar_msg_type, args.lidar_topic, self.lidar_callback, qos)

        self.create_service(PanoBBoxDepth, args.depth_service, self.handle_depth)
        self.create_service(PanoBBoxPosition, args.position_service, self.handle_position)
        self.create_service(PanoBBoxBearing, args.bearing_service, self.handle_bearing)
        self.selected_points_publisher = self.create_publisher(
            PointCloud2,
            args.selected_points_topic,
            args.queue_depth,
        )
        self.used_points_publisher = self.create_publisher(
            PointCloud2,
            args.used_points_topic,
            args.queue_depth,
        )
        self.outlier_points_publisher = self.create_publisher(
            PointCloud2,
            args.outlier_points_topic,
            args.queue_depth,
        )
        self.result_marker_publisher = self.create_publisher(
            Marker,
            args.result_marker_topic,
            args.queue_depth,
        )

        self.get_logger().info("Subscribing to {}".format(args.lidar_topic))
        self.get_logger().info("Camera frame: {}".format(args.camera_frame))
        self.get_logger().info("Default output frame: {}".format(args.default_frame))
        self.get_logger().info("Services: {}, {}, {}".format(args.depth_service, args.position_service, args.bearing_service))
        self.get_logger().info("Publishing bbox-selected points to {}".format(args.selected_points_topic))
        self.get_logger().info("Publishing used points to {}".format(args.used_points_topic))
        self.get_logger().info("Publishing outlier points to {}".format(args.outlier_points_topic))
        self.get_logger().info("Publishing result marker to {}".format(args.result_marker_topic))

    def resolve_topic_type(self, topic_name):
        deadline = self.get_clock().now() + Duration(seconds=self.args.topic_wait_timeout)
        while rclpy.ok() and self.get_clock().now() < deadline:
            for name, topic_types in self.get_topic_names_and_types():
                if name == topic_name and topic_types:
                    return get_message(topic_types[0])
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("Could not determine topic type for {}".format(topic_name))

    def lidar_callback(self, msg):
        source_frame = getattr(getattr(msg, "header", None), "frame_id", "") or self.args.source_frame
        stamp = getattr(getattr(msg, "header", None), "stamp", rclpy.time.Time().to_msg())

        try:
            points_lidar = xyz_array_from_lidar_msg(msg)
            transform = self.tf_buffer.lookup_transform(
                self.args.camera_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=self.args.tf_timeout),
            )
            points_camera = transform_points(points_lidar, transform).astype(np.float32)
            with self.lock:
                self.latest_points_camera = points_camera
                self.latest_stamp = stamp
                self.latest_input_count = len(points_lidar)
        except Exception as exc:
            self.get_logger().warn("Could not cache lidar cloud: {}".format(exc))

    def handle_depth(self, request, response):
        result = self.run_inference(request)
        response.valid = result.valid
        response.message = result.message
        response.input_count = result.input_count
        response.selected_count = result.selected_count
        response.used_count = result.used_count
        response.depth_m = result.depth_m
        return response

    def handle_position(self, request, response):
        result = self.run_inference(request)
        fill_common_spatial_response(response, result)
        return response

    def handle_bearing(self, request, response):
        result = self.run_inference(request)
        fill_common_spatial_response(response, result)
        response.xy_distance_m = result.xy_distance_m
        response.angle_rad = result.angle_rad
        response.angle_deg = math.degrees(result.angle_rad) if result.point is not None else float("nan")
        return response

    def run_inference(self, request) -> InferenceResult:
        with self.lock:
            points_camera = None if self.latest_points_camera is None else self.latest_points_camera.copy()
            stamp = self.latest_stamp
            input_count = self.latest_input_count

        output_frame = request.frame_id.strip() or self.args.default_frame
        if points_camera is None or stamp is None:
            self.publish_result_marker(None, output_frame, stamp)
            return self.invalid("No lidar cloud has been cached yet", input_count, output_frame, stamp)

        if request.image_width == 0 or request.image_height == 0:
            self.publish_result_marker(None, output_frame, stamp)
            return self.invalid("image_width and image_height must be non-zero", input_count, output_frame, stamp)

        bbox = (request.x_min, request.y_min, request.x_max, request.y_max)
        bounds = pixel_bbox_to_angle_bounds(bbox, request.image_width, request.image_height)
        selected_camera = points_camera[points_inside_camera_bounds(points_camera, self.rotation, bounds)]
        selected_count = len(selected_camera)
        if selected_count == 0:
            self.publish_result_marker(None, output_frame, stamp)
            return self.invalid("No lidar points inside bbox", input_count, output_frame, stamp, selected_count=0)

        try:
            points_out = self.transform_camera_points_to_output(selected_camera, output_frame, stamp)
        except Exception as exc:
            self.publish_result_marker(None, output_frame, stamp)
            return self.invalid(str(exc), input_count, output_frame, stamp, selected_count=selected_count)
        self.publish_points(self.selected_points_publisher, points_out, output_frame, stamp)

        keep_mask = outlier_keep_mask(
            points_out,
            request.outlier_filter.strip() or self.args.outlier_filter,
            self.args.mad_threshold,
            self.args.statistical_std_ratio,
            self.args.radius,
            self.args.radius_min_neighbors,
        )
        filtered = points_out[keep_mask]
        filtered_camera = selected_camera[keep_mask]
        outliers = points_out[~keep_mask]
        self.publish_points(self.outlier_points_publisher, outliers, output_frame, stamp)
        if len(filtered) == 0:
            self.publish_points(self.used_points_publisher, filtered, output_frame, stamp)
            self.publish_result_marker(None, output_frame, stamp)
            return self.invalid(
                "Outlier filter removed every selected point",
                input_count,
                output_frame,
                stamp,
                selected_count=selected_count,
            )

        method = request.method.strip() or self.args.method
        point, used_points = reduce_points(
            filtered,
            method,
            int(request.k or self.args.k),
            self.args.cluster_tolerance,
            self.args.min_cluster_size,
            bbox_center_ray(bounds, self.rotation),
            points_camera=filtered_camera,
        )
        if point is None or used_points is None or len(used_points) == 0:
            self.publish_points(self.used_points_publisher, np.empty((0, 3), dtype=np.float32), output_frame, stamp)
            self.publish_result_marker(None, output_frame, stamp)
            return self.invalid(
                "Could not compute representative point",
                input_count,
                output_frame,
                stamp,
                selected_count=selected_count,
            )
        self.publish_points(self.used_points_publisher, used_points, output_frame, stamp)
        self.publish_result_marker(point, output_frame, stamp)

        return InferenceResult(
            valid=True,
            message="ok",
            input_count=input_count,
            selected_count=selected_count,
            used_count=len(used_points),
            point=point,
            stamp=stamp,
            frame_id=output_frame,
        )

    def transform_camera_points_to_output(self, points_camera, output_frame, stamp):
        if output_frame == self.args.camera_frame:
            return points_camera
        transform = self.tf_buffer.lookup_transform(
            output_frame,
            self.args.camera_frame,
            stamp,
            timeout=Duration(seconds=self.args.tf_timeout),
        )
        return transform_points(points_camera, transform).astype(np.float32)

    def publish_points(self, publisher, points_xyz, frame_id, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        points = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
        cloud = point_cloud2.create_cloud_xyz32(header, points.tolist())
        publisher.publish(cloud)

    def publish_result_marker(self, point, frame_id, stamp):
        marker = Marker()
        if stamp is not None:
            marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "bbox_result"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.DELETE if point is None else Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.args.marker_scale
        marker.scale.y = self.args.marker_scale
        marker.scale.z = self.args.marker_scale
        marker.color.r = 1.0
        marker.color.g = 0.25
        marker.color.b = 0.0
        marker.color.a = 0.95
        if point is not None:
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = float(point[2])
        self.result_marker_publisher.publish(marker)

    @staticmethod
    def invalid(message, input_count, frame_id, stamp, selected_count=0):
        return InferenceResult(
            valid=False,
            message=message,
            input_count=input_count,
            selected_count=selected_count,
            used_count=0,
            point=None,
            stamp=stamp,
            frame_id=frame_id,
        )


def fill_common_spatial_response(response, result):
    response.valid = result.valid
    response.message = result.message
    response.input_count = result.input_count
    response.selected_count = result.selected_count
    response.used_count = result.used_count
    response.depth_m = result.depth_m
    response.point = make_point_stamped(result.point, result.frame_id, result.stamp)


def make_point_stamped(point, frame_id, stamp):
    msg = PointStamped()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    if point is not None:
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
    return msg


def load_calibration_rotation(path):
    path = Path(path).expanduser()
    yaw = -1.0
    pitch = -6.0
    roll = -180.0
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        yaw = float(payload.get("yaw", yaw))
        pitch = float(payload.get("pitch", pitch))
        roll = float(payload.get("roll", roll))
    return euler_zyx_degrees_to_matrix(yaw, pitch, roll)


def pixel_bbox_to_angle_bounds(bbox, width, height):
    x0, y0, x1, y1 = bbox
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    yaw_min = x_min / float(width) * 360.0 - 180.0
    yaw_max = x_max / float(width) * 360.0 - 180.0
    elevation_min = y_min / float(height) * 180.0 - 90.0
    elevation_max = y_max / float(height) * 180.0 - 90.0
    return (yaw_min, yaw_max, elevation_min, elevation_max)


def points_inside_camera_bounds(points_camera_frame, image_rotation, angle_bounds):
    yaw_min, yaw_max, elevation_min, elevation_max = angle_bounds
    if len(points_camera_frame) == 0:
        return np.zeros((0,), dtype=bool)

    norms = np.linalg.norm(points_camera_frame, axis=1)
    valid = norms > 1e-6
    directions_frame = np.zeros_like(points_camera_frame, dtype=np.float32)
    directions_frame[valid] = points_camera_frame[valid] / norms[valid, None]

    directions_camera = directions_frame @ image_rotation
    yaw = np.rad2deg(np.arctan2(directions_camera[:, 1], directions_camera[:, 0]))
    elevation = np.rad2deg(
        np.arctan2(directions_camera[:, 2], np.hypot(directions_camera[:, 0], directions_camera[:, 1]))
    )
    return valid & (yaw >= yaw_min) & (yaw <= yaw_max) & (elevation >= elevation_min) & (elevation <= elevation_max)


def euler_zyx_degrees_to_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return rz @ ry @ rx


def bbox_center_ray(angle_bounds, image_rotation):
    yaw_min, yaw_max, elevation_min, elevation_max = angle_bounds
    yaw = math.radians((yaw_min + yaw_max) * 0.5)
    elevation = math.radians((elevation_min + elevation_max) * 0.5)
    ray_camera = np.array(
        [
            math.cos(elevation) * math.cos(yaw),
            math.cos(elevation) * math.sin(yaw),
            math.sin(elevation),
        ],
        dtype=np.float32,
    )
    return ray_camera @ image_rotation.T


def reduce_points(points, method, k, cluster_tolerance, min_cluster_size, center_ray_camera, points_camera):
    method = method.lower()
    if method == METHOD_ALL:
        return np.mean(points, axis=0), points
    if method == METHOD_CENTER_K:
        return reduce_center_k(points, max(1, k), center_ray_camera, points_camera)
    if method == METHOD_MAIN_CLUSTER:
        labels = euclidean_clusters(points, cluster_tolerance)
        if labels.size == 0:
            return None, None
        label, count = largest_cluster_label(labels)
        if label < 0 or count < min_cluster_size:
            return None, None
        cluster_points = points[labels == label]
        return np.mean(cluster_points, axis=0), cluster_points
    raise ValueError("Unknown method '{}'; use center_k, all, or main_cluster".format(method))


def reduce_center_k(points_out, k, center_ray_camera, points_camera):
    norms = np.linalg.norm(points_camera, axis=1)
    valid = norms > 1e-6
    if not np.any(valid):
        return None, None
    directions = np.zeros_like(points_camera, dtype=np.float32)
    directions[valid] = points_camera[valid] / norms[valid, None]
    scores = directions @ center_ray_camera
    scores[~valid] = -np.inf
    indices = np.argsort(-scores)[: min(k, len(points_out))]
    used_points = points_out[indices]
    return np.mean(used_points, axis=0), used_points


def outlier_keep_mask(points, mode, mad_threshold, statistical_std_ratio, radius, radius_min_neighbors):
    mode = mode.lower()
    if mode == OUTLIER_NONE:
        return np.ones((len(points),), dtype=bool)
    if mode == OUTLIER_MAD_RANGE:
        return mad_range_keep_mask(points, mad_threshold)
    if mode == OUTLIER_STATISTICAL:
        return statistical_keep_mask(points, statistical_std_ratio)
    if mode == OUTLIER_RADIUS:
        return radius_keep_mask(points, radius, radius_min_neighbors)
    raise ValueError("Unknown outlier_filter '{}'; use none, mad_range, statistical, or radius".format(mode))


def mad_range_keep_mask(points, threshold):
    ranges = np.linalg.norm(points, axis=1)
    median = np.median(ranges)
    mad = np.median(np.abs(ranges - median))
    if mad < 1e-6:
        return np.ones((len(points),), dtype=bool)
    robust_z = 0.6745 * np.abs(ranges - median) / mad
    return robust_z <= threshold


def statistical_keep_mask(points, std_ratio):
    if len(points) < 4:
        return np.ones((len(points),), dtype=bool)
    distances = pairwise_distances(points)
    nearest_mean = np.mean(np.sort(distances, axis=1)[:, 1 : min(9, len(points))], axis=1)
    limit = float(np.mean(nearest_mean) + std_ratio * np.std(nearest_mean))
    return nearest_mean <= limit


def radius_keep_mask(points, radius, min_neighbors):
    if len(points) < 2:
        return np.ones((len(points),), dtype=bool)
    distances = pairwise_distances(points)
    neighbor_counts = np.sum(distances <= radius, axis=1) - 1
    return neighbor_counts >= min_neighbors


def euclidean_clusters(points, tolerance):
    if len(points) == 0:
        return np.zeros((0,), dtype=np.int32)
    distances = pairwise_distances(points)
    labels = np.full((len(points),), -1, dtype=np.int32)
    cluster_id = 0
    for start in range(len(points)):
        if labels[start] >= 0:
            continue
        queue = [start]
        labels[start] = cluster_id
        while queue:
            idx = queue.pop()
            neighbors = np.flatnonzero((distances[idx] <= tolerance) & (labels < 0))
            labels[neighbors] = cluster_id
            queue.extend(neighbors.tolist())
        cluster_id += 1
    return labels


def largest_cluster_label(labels):
    valid = labels[labels >= 0]
    if valid.size == 0:
        return -1, 0
    counts = np.bincount(valid)
    label = int(np.argmax(counts))
    return label, int(counts[label])


def pairwise_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve bbox-based depth, position, and bearing estimates.")
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--source-frame", default=DEFAULT_SOURCE_FRAME)
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--default-frame", default=DEFAULT_OUTPUT_FRAME)
    parser.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--depth-service", default="/bbox_depth")
    parser.add_argument("--position-service", default="/bbox_position")
    parser.add_argument("--bearing-service", default="/bbox_bearing")
    parser.add_argument("--selected-points-topic", default=DEFAULT_SELECTED_POINTS_TOPIC)
    parser.add_argument("--used-points-topic", default=DEFAULT_USED_POINTS_TOPIC)
    parser.add_argument("--outlier-points-topic", default=DEFAULT_OUTLIER_POINTS_TOPIC)
    parser.add_argument("--result-marker-topic", default=DEFAULT_RESULT_MARKER_TOPIC)
    parser.add_argument("--marker-scale", type=float, default=0.18)
    parser.add_argument("--method", default=METHOD_CENTER_K, choices=[METHOD_CENTER_K, METHOD_ALL, METHOD_MAIN_CLUSTER])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--outlier-filter", default=OUTLIER_MAD_RANGE, choices=[OUTLIER_NONE, OUTLIER_MAD_RANGE, OUTLIER_STATISTICAL, OUTLIER_RADIUS])
    parser.add_argument("--mad-threshold", type=float, default=3.5)
    parser.add_argument("--statistical-std-ratio", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--radius-min-neighbors", type=int, default=2)
    parser.add_argument("--cluster-tolerance", type=float, default=0.35)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--queue-depth", type=int, default=10)
    parser.add_argument("--tf-timeout", type=float, default=0.05)
    parser.add_argument("--topic-wait-timeout", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = BBoxInferenceNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

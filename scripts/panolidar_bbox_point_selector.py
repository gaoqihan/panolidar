#!/usr/bin/env python3
"""
Select lidar points by drawing a bounding box on the 360 camera image.

The saved calibration from panolidar_calibration_viewer.py maps each image
pixel ray into camera_360_link. The selected lidar points are published as PointCloud2.
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

qt_plugin_path = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
if os.path.isdir(qt_plugin_path):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path
os.environ.pop("QT_PLUGIN_PATH", None)

import numpy as np
import rclpy
import tf2_ros
from PyQt5 import QtCore, QtGui, QtWidgets
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from spherical_lidar_camera import transform_points, xyz_array_from_lidar_msg


DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_IMAGE_TOPIC = "/image_raw"
DEFAULT_OUTPUT_TOPIC = "/vis_selected_points"
DEFAULT_TARGET_FRAME = "z1_link"
DEFAULT_SOURCE_FRAME = "livox_link"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "config" / "panolidar_calibration_params.json"


class BboxSelectionNode(Node):
    def __init__(self, args):
        super().__init__("panolidar_image_bbox_point_selector")
        self.args = args
        self.lock = threading.Lock()
        self.latest_image_rgb = None
        self.image_version = 0
        self.image_size = None
        self.bbox = None
        self.angle_bounds = None
        self.selected_count = 0
        self.input_count = 0
        self.last_status = "waiting for image/lidar"
        self.last_warn_time = None
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
        self.create_subscription(Image, args.image_topic, self.image_callback, qos)
        self.publisher = self.create_publisher(PointCloud2, args.output_topic, args.queue_depth)

        self.get_logger().info("Lidar topic: {}".format(args.lidar_topic))
        self.get_logger().info("Image topic: {}".format(args.image_topic))
        self.get_logger().info("Publishing selected points to {}".format(args.output_topic))
        self.get_logger().info("Loaded calibration from {}".format(args.calibration_path))

    def resolve_topic_type(self, topic_name):
        deadline = self.get_clock().now() + Duration(seconds=self.args.topic_wait_timeout)
        while rclpy.ok() and self.get_clock().now() < deadline:
            for name, topic_types in self.get_topic_names_and_types():
                if name == topic_name and topic_types:
                    return get_message(topic_types[0])
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("Could not determine topic type for {}".format(topic_name))

    def image_callback(self, msg):
        try:
            image = ros_image_to_rgb(msg)
            with self.lock:
                self.latest_image_rgb = image
                self.image_size = (image.shape[1], image.shape[0])
                self.image_version += 1
        except Exception as exc:
            self.set_status(str(exc))

    def lidar_callback(self, msg):
        with self.lock:
            angle_bounds = self.angle_bounds

        if angle_bounds is None:
            return

        source_frame = getattr(getattr(msg, "header", None), "frame_id", "") or self.args.source_frame
        stamp = getattr(getattr(msg, "header", None), "stamp", rclpy.time.Time().to_msg())

        try:
            points_lidar = xyz_array_from_lidar_msg(msg)
            transform = self.tf_buffer.lookup_transform(
                self.args.target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=self.args.tf_timeout),
            )
            points_z1 = transform_points(points_lidar, transform).astype(np.float32)
            mask = points_inside_camera_bounds(points_z1, self.rotation, angle_bounds)
            selected = points_z1[mask]
            self.publish_points(selected, stamp)

            with self.lock:
                self.input_count = len(points_z1)
                self.selected_count = len(selected)
                self.last_status = "publishing"
        except Exception as exc:
            self.set_status(str(exc))

    def set_bbox(self, bbox):
        with self.lock:
            self.bbox = bbox
            image_size = self.image_size

        if bbox is None or image_size is None:
            with self.lock:
                self.angle_bounds = None
            return

        bounds = pixel_bbox_to_angle_bounds(bbox, image_size[0], image_size[1])
        with self.lock:
            self.angle_bounds = bounds

    def snapshot(self):
        with self.lock:
            image = None if self.latest_image_rgb is None else self.latest_image_rgb.copy()
            image_version = self.image_version
            bbox = self.bbox
            bounds = self.angle_bounds
            selected_count = self.selected_count
            input_count = self.input_count
            status = self.last_status
        return image, image_version, bbox, bounds, selected_count, input_count, status

    def publish_points(self, points_xyz, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.args.target_frame
        cloud = point_cloud2.create_cloud_xyz32(header, np.asarray(points_xyz, dtype=np.float32).tolist())
        self.publisher.publish(cloud)

    def set_status(self, status):
        with self.lock:
            self.last_status = status

        now = self.get_clock().now()
        if self.last_warn_time is None or (now - self.last_warn_time).nanoseconds > 2_000_000_000:
            self.get_logger().warn(status)
            self.last_warn_time = now


class ImageBboxWidget(QtWidgets.QWidget):
    bbox_changed = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.image = None
        self.qimage = None
        self.bbox = None
        self.drag_start = None
        self.image_rect = QtCore.QRectF()
        self.setMinimumSize(900, 450)
        self.setMouseTracking(True)

    def set_image(self, image_rgb):
        self.image = image_rgb
        height, width = image_rgb.shape[:2]
        self.qimage = QtGui.QImage(
            image_rgb.data,
            width,
            height,
            width * 3,
            QtGui.QImage.Format_RGB888,
        ).copy()
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(18, 20, 24))
        if self.qimage is None:
            painter.setPen(QtGui.QColor(230, 230, 230))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "waiting for image")
            return

        self.image_rect = scaled_rect(self.qimage.width(), self.qimage.height(), self.rect())
        painter.drawImage(self.image_rect, self.qimage)

        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            rect = self.image_bbox_to_widget_rect(x0, y0, x1, y1)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 80), 3))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if self.qimage is None or event.button() != QtCore.Qt.LeftButton:
            return
        point = self.widget_to_image_point(event.pos())
        if point is None:
            return
        self.drag_start = point
        self.bbox = (point[0], point[1], point[0], point[1])
        self.bbox_changed.emit(self.normalized_bbox())
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag_start is None:
            return
        point = self.widget_to_image_point(event.pos(), clamp=True)
        self.bbox = (self.drag_start[0], self.drag_start[1], point[0], point[1])
        self.bbox_changed.emit(self.normalized_bbox())
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.drag_start is not None:
            point = self.widget_to_image_point(event.pos(), clamp=True)
            self.bbox = (self.drag_start[0], self.drag_start[1], point[0], point[1])
            self.drag_start = None
            self.bbox_changed.emit(self.normalized_bbox())
            self.update()

    def clear_bbox(self):
        self.bbox = None
        self.drag_start = None
        self.bbox_changed.emit(None)
        self.update()

    def normalized_bbox(self):
        if self.bbox is None:
            return None
        x0, y0, x1, y1 = self.bbox
        x_min, x_max = sorted((int(round(x0)), int(round(x1))))
        y_min, y_max = sorted((int(round(y0)), int(round(y1))))
        if x_max <= x_min or y_max <= y_min:
            return None
        return (x_min, y_min, x_max, y_max)

    def widget_to_image_point(self, pos, clamp=False):
        if self.qimage is None or self.image_rect.isNull():
            return None
        x = pos.x()
        y = pos.y()
        if not self.image_rect.contains(QtCore.QPointF(x, y)) and not clamp:
            return None
        x = min(max(x, self.image_rect.left()), self.image_rect.right())
        y = min(max(y, self.image_rect.top()), self.image_rect.bottom())
        u = (x - self.image_rect.left()) / self.image_rect.width() * self.qimage.width()
        v = (y - self.image_rect.top()) / self.image_rect.height() * self.qimage.height()
        return (u, v)

    def image_bbox_to_widget_rect(self, x0, y0, x1, y1):
        sx = self.image_rect.width() / self.qimage.width()
        sy = self.image_rect.height() / self.qimage.height()
        left = self.image_rect.left() + min(x0, x1) * sx
        right = self.image_rect.left() + max(x0, x1) * sx
        top = self.image_rect.top() + min(y0, y1) * sy
        bottom = self.image_rect.top() + max(y0, y1) * sy
        return QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom))


class BboxSelectorWindow(QtWidgets.QMainWindow):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.last_image_version = -1
        self.setWindowTitle("360 camera image bbox to lidar point selector")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.image_widget = ImageBboxWidget()
        self.image_widget.bbox_changed.connect(self.node.set_bbox)
        layout.addWidget(self.image_widget, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("waiting")
        clear_button = QtWidgets.QPushButton("Clear bbox")
        clear_button.clicked.connect(self.image_widget.clear_bbox)
        row.addWidget(self.status_label, stretch=1)
        row.addWidget(clear_button)
        layout.addLayout(row)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_from_node)
        self.timer.start(50)

    def update_from_node(self):
        image, image_version, _bbox, bounds, selected_count, input_count, status = self.node.snapshot()
        if image is not None and image_version != self.last_image_version:
            self.image_widget.set_image(image)
            self.last_image_version = image_version

        if bounds is None:
            bounds_text = "draw bbox"
        else:
            bounds_text = "yaw {:+.1f}..{:+.1f}, elev {:+.1f}..{:+.1f}".format(*bounds)
        self.status_label.setText(
            "{} | selected {} / {} | {}".format(bounds_text, selected_count, input_count, status)
        )

    def closeEvent(self, event):
        if rclpy.ok():
            rclpy.shutdown()
        event.accept()


def scaled_rect(image_width, image_height, outer_rect):
    image_aspect = image_width / float(image_height)
    outer_aspect = outer_rect.width() / float(max(1, outer_rect.height()))
    if outer_aspect > image_aspect:
        height = outer_rect.height()
        width = height * image_aspect
    else:
        width = outer_rect.width()
        height = width / image_aspect
    left = outer_rect.left() + (outer_rect.width() - width) / 2.0
    top = outer_rect.top() + (outer_rect.height() - height) / 2.0
    return QtCore.QRectF(left, top, width, height)


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


def points_inside_camera_bounds(points_z1, image_rotation, angle_bounds):
    yaw_min, yaw_max, elevation_min, elevation_max = angle_bounds
    if len(points_z1) == 0:
        return np.zeros((0,), dtype=bool)

    norms = np.linalg.norm(points_z1, axis=1)
    valid = norms > 1e-6
    directions_z1 = np.zeros_like(points_z1, dtype=np.float32)
    directions_z1[valid] = points_z1[valid] / norms[valid, None]

    directions_camera = directions_z1 @ image_rotation
    yaw = np.rad2deg(np.arctan2(directions_camera[:, 1], directions_camera[:, 0]))
    elevation = np.rad2deg(np.arctan2(directions_camera[:, 2], np.hypot(directions_camera[:, 0], directions_camera[:, 1])))
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


def ros_image_to_rgb(msg):
    encoding = msg.encoding.lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError("Unsupported image encoding: {}".format(msg.encoding))

    channels = channels_by_encoding[encoding]
    row = np.frombuffer(msg.data, dtype=np.uint8)
    expected_size = int(msg.step) * int(msg.height)
    if row.size < expected_size:
        raise ValueError("Image data is smaller than height * step")

    rows = row[:expected_size].reshape((msg.height, msg.step))
    pixels = rows[:, : msg.width * channels].reshape((msg.height, msg.width, channels))

    if encoding in ("rgb8", "8uc3"):
        return np.ascontiguousarray(pixels[:, :, :3])
    if encoding == "bgr8":
        return np.ascontiguousarray(pixels[:, :, ::-1])
    if encoding == "rgba8":
        return np.ascontiguousarray(pixels[:, :, :3])
    if encoding == "bgra8":
        return np.ascontiguousarray(pixels[:, :, [2, 1, 0]])
    if encoding in ("mono8", "8uc1"):
        return np.ascontiguousarray(np.repeat(pixels, 3, axis=2))
    if encoding == "8uc4":
        return np.ascontiguousarray(pixels[:, :, :3])

    raise ValueError("Unsupported image encoding: {}".format(msg.encoding))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a bbox on the 360 camera image and publish matching lidar points."
    )
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--output-topic", default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--target-frame", default=DEFAULT_TARGET_FRAME)
    parser.add_argument("--source-frame", default=DEFAULT_SOURCE_FRAME)
    parser.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--queue-depth", type=int, default=10)
    parser.add_argument("--tf-timeout", type=float, default=0.05)
    parser.add_argument("--topic-wait-timeout", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication([])
    rclpy.init()
    node = BboxSelectionNode(args)
    window = BboxSelectorWindow(node)
    window.resize(1280, 720)
    window.show()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        app.exec_()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == "__main__":
    main()

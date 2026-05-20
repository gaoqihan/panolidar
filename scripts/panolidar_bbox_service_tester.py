#!/usr/bin/env python3
"""
PyQt interface for testing bbox inference services.

Draw a bbox on the 360 camera image, choose one of the bbox inference services,
pick aggregation/outlier options, and inspect the returned value plus service
round-trip time.
"""

import argparse
import os
import threading
import time

qt_plugin_path = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
if os.path.isdir(qt_plugin_path):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path
os.environ.pop("QT_PLUGIN_PATH", None)

import numpy as np
import rclpy
from PyQt5 import QtCore, QtGui, QtWidgets
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from panolidar.srv import PanoBBoxBearing, PanoBBoxDepth, PanoBBoxPosition


DEFAULT_IMAGE_TOPIC = "/image_raw"
DEFAULT_DEPTH_SERVICE = "/bbox_depth"
DEFAULT_POSITION_SERVICE = "/bbox_position"
DEFAULT_BEARING_SERVICE = "/bbox_bearing"
DEFAULT_FRAME = "base_link"


class SignalBridge(QtCore.QObject):
    result_ready = QtCore.pyqtSignal(str)
    status_ready = QtCore.pyqtSignal(str)


class BBoxServiceTesterNode(Node):
    def __init__(self, args, bridge):
        super().__init__("panolidar_bbox_service_tester")
        self.args = args
        self.bridge = bridge
        self.lock = threading.Lock()
        self.latest_image_rgb = None
        self.image_version = 0
        self.image_size = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=args.queue_depth,
        )
        self.create_subscription(Image, args.image_topic, self.image_callback, qos)
        self.depth_client = self.create_client(PanoBBoxDepth, args.depth_service)
        self.position_client = self.create_client(PanoBBoxPosition, args.position_service)
        self.bearing_client = self.create_client(PanoBBoxBearing, args.bearing_service)

        self.get_logger().info("Image topic: {}".format(args.image_topic))
        self.get_logger().info(
            "Services: {}, {}, {}".format(args.depth_service, args.position_service, args.bearing_service)
        )

    def image_callback(self, msg):
        try:
            image = ros_image_to_rgb(msg)
            with self.lock:
                self.latest_image_rgb = image
                self.image_size = (image.shape[1], image.shape[0])
                self.image_version += 1
        except Exception as exc:
            self.bridge.status_ready.emit(str(exc))

    def snapshot(self):
        with self.lock:
            image = None if self.latest_image_rgb is None else self.latest_image_rgb.copy()
            return image, self.image_version, self.image_size

    def call_service(self, service_kind, bbox, image_size, frame_id, method, k, outlier_filter):
        if bbox is None:
            self.bridge.result_ready.emit("Draw a bbox first.")
            return
        if image_size is None:
            self.bridge.result_ready.emit("Waiting for image size.")
            return

        service_kind = service_kind.lower()
        if service_kind == "depth":
            client = self.depth_client
            request = PanoBBoxDepth.Request()
        elif service_kind == "position":
            client = self.position_client
            request = PanoBBoxPosition.Request()
        elif service_kind == "bearing":
            client = self.bearing_client
            request = PanoBBoxBearing.Request()
        else:
            self.bridge.result_ready.emit("Unknown service kind: {}".format(service_kind))
            return

        if not client.service_is_ready():
            self.bridge.result_ready.emit("{} service is not ready yet.".format(service_kind))
            return

        x_min, y_min, x_max, y_max = bbox
        request.x_min = float(x_min)
        request.y_min = float(y_min)
        request.x_max = float(x_max)
        request.y_max = float(y_max)
        request.image_width = int(image_size[0])
        request.image_height = int(image_size[1])
        request.frame_id = frame_id.strip()
        request.method = method
        request.k = int(k)
        request.outlier_filter = outlier_filter

        started = time.perf_counter_ns()
        future = client.call_async(request)
        future.add_done_callback(
            lambda done_future: self._handle_response(service_kind, done_future, started, bbox, image_size)
        )
        self.bridge.status_ready.emit("Request sent to {}.".format(service_kind))

    def _handle_response(self, service_kind, future, started_ns, bbox, image_size):
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        try:
            response = future.result()
        except Exception as exc:
            self.bridge.result_ready.emit(
                "{} failed after {:.2f} ms:\n{}".format(service_kind, elapsed_ms, exc)
            )
            return

        self.bridge.result_ready.emit(format_response(service_kind, response, elapsed_ms, bbox, image_size))
        self.bridge.status_ready.emit("Received {} result in {:.2f} ms.".format(service_kind, elapsed_ms))


class ImageBboxWidget(QtWidgets.QWidget):
    bbox_changed = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.image = None
        self.qimage = None
        self.bbox = None
        self.drag_start = None
        self.image_rect = QtCore.QRectF()
        self.setMinimumSize(860, 480)
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


class BBoxServiceTesterWindow(QtWidgets.QMainWindow):
    def __init__(self, node, bridge):
        super().__init__()
        self.node = node
        self.bridge = bridge
        self.last_image_version = -1
        self.image_size = None
        self.bbox = None
        self.setWindowTitle("360 camera bbox service tester")

        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        self.setCentralWidget(central)

        self.image_widget = ImageBboxWidget()
        self.image_widget.bbox_changed.connect(self.on_bbox_changed)
        root.addWidget(self.image_widget, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(360)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)
        root.addWidget(panel)

        self.status_label = QtWidgets.QLabel("waiting for image")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        panel_layout.addLayout(form)

        self.frame_edit = QtWidgets.QLineEdit(DEFAULT_FRAME)
        form.addRow("Frame", self.frame_edit)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["center_k", "all", "main_cluster"])
        form.addRow("Method", self.method_combo)

        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 500)
        self.k_spin.setValue(10)
        form.addRow("k", self.k_spin)

        self.outlier_combo = QtWidgets.QComboBox()
        self.outlier_combo.addItems(["mad_range", "none", "statistical", "radius"])
        form.addRow("Outliers", self.outlier_combo)

        self.bbox_label = QtWidgets.QLabel("bbox: none")
        self.bbox_label.setWordWrap(True)
        panel_layout.addWidget(self.bbox_label)

        button_row = QtWidgets.QHBoxLayout()
        self.depth_button = QtWidgets.QPushButton("Depth")
        self.position_button = QtWidgets.QPushButton("Position")
        self.bearing_button = QtWidgets.QPushButton("Bearing")
        button_row.addWidget(self.depth_button)
        button_row.addWidget(self.position_button)
        button_row.addWidget(self.bearing_button)
        panel_layout.addLayout(button_row)

        self.depth_button.clicked.connect(lambda: self.call_service("depth"))
        self.position_button.clicked.connect(lambda: self.call_service("position"))
        self.bearing_button.clicked.connect(lambda: self.call_service("bearing"))

        self.clear_button = QtWidgets.QPushButton("Clear bbox")
        self.clear_button.clicked.connect(self.image_widget.clear_bbox)
        panel_layout.addWidget(self.clear_button)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(280)
        self.output.setPlainText("Draw a bbox, choose options, then press a service button.")
        panel_layout.addWidget(self.output, stretch=1)

        self.bridge.result_ready.connect(self.output.setPlainText)
        self.bridge.status_ready.connect(self.status_label.setText)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_from_node)
        self.timer.start(50)

    def on_bbox_changed(self, bbox):
        self.bbox = bbox
        if bbox is None:
            self.bbox_label.setText("bbox: none")
        else:
            self.bbox_label.setText("bbox: x {}..{}, y {}..{}".format(bbox[0], bbox[2], bbox[1], bbox[3]))

    def update_from_node(self):
        image, image_version, image_size = self.node.snapshot()
        self.image_size = image_size
        if image is not None and image_version != self.last_image_version:
            self.image_widget.set_image(image)
            self.last_image_version = image_version
        if image_size is not None and self.bbox is None:
            self.status_label.setText("image {}x{} | draw bbox".format(image_size[0], image_size[1]))

    def call_service(self, service_kind):
        self.node.call_service(
            service_kind,
            self.bbox,
            self.image_size,
            self.frame_edit.text(),
            self.method_combo.currentText(),
            self.k_spin.value(),
            self.outlier_combo.currentText(),
        )

    def closeEvent(self, event):
        if rclpy.ok():
            rclpy.shutdown()
        event.accept()


def format_response(service_kind, response, elapsed_ms, bbox, image_size):
    lines = [
        "service: {}".format(service_kind),
        "round trip: {:.2f} ms".format(elapsed_ms),
        "valid: {}".format(response.valid),
        "message: {}".format(response.message),
        "bbox: x {}..{}, y {}..{} on {}x{}".format(
            bbox[0], bbox[2], bbox[1], bbox[3], image_size[0], image_size[1]
        ),
        "counts: input={}, selected={}, used={}".format(
            response.input_count,
            response.selected_count,
            response.used_count,
        ),
        "depth_m: {:.4f}".format(response.depth_m),
    ]

    if hasattr(response, "point"):
        point = response.point.point
        lines.extend(
            [
                "frame: {}".format(response.point.header.frame_id),
                "point: x={:.4f}, y={:.4f}, z={:.4f}".format(point.x, point.y, point.z),
            ]
        )
    if hasattr(response, "xy_distance_m"):
        lines.extend(
            [
                "xy_distance_m: {:.4f}".format(response.xy_distance_m),
                "angle_rad: {:.4f}".format(response.angle_rad),
                "angle_deg: {:.2f}".format(response.angle_deg),
            ]
        )

    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(description="Draw bboxes and test bbox inference services.")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--depth-service", default=DEFAULT_DEPTH_SERVICE)
    parser.add_argument("--position-service", default=DEFAULT_POSITION_SERVICE)
    parser.add_argument("--bearing-service", default=DEFAULT_BEARING_SERVICE)
    parser.add_argument("--queue-depth", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication([])
    rclpy.init()
    bridge = SignalBridge()
    node = BBoxServiceTesterNode(args, bridge)
    window = BBoxServiceTesterWindow(node, bridge)
    window.resize(1320, 760)
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

#!/usr/bin/env python3
"""
Live lidar/360-camera angular calibration viewer.

This script shows:
  - /lidar_360/lidar transformed into camera_360_link as a 3D point cloud
  - a rectangular patch from the 360 camera stitched/equirectangular image
  - four rays from the origin through the patch corners, extended into the cloud
  - sliders for rotating the image sphere in yaw/pitch/roll

The slider values are the first parameters to tune for:
  pixel coordinate -> spherical angle -> camera_360_link direction
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
import vtk
from PIL import Image as PilImage
from PyQt5 import QtCore, QtWidgets
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support

from spherical_lidar_camera import transform_points, xyz_array_from_lidar_msg


DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_IMAGE_TOPIC = "/image_raw"
DEFAULT_TARGET_FRAME = "z1_link"
DEFAULT_SOURCE_FRAME = "livox_link"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "config" / "panolidar_calibration_params.json"


class RosDataBridge(Node):
    def __init__(self, args):
        super().__init__("panolidar_360_lidar_calibration_viewer")
        self.args = args
        self.lock = threading.Lock()
        self.latest_points_z1 = np.zeros((0, 3), dtype=np.float32)
        self.latest_image_rgb = None
        self.image_version = 0
        self.last_lidar_time = 0.0
        self.last_image_time = 0.0
        self.last_status = "waiting for lidar/image"
        self.last_warn_time = None

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

        self.get_logger().info("Lidar topic: {}".format(args.lidar_topic))
        self.get_logger().info("Image topic: {}".format(args.image_topic))
        self.get_logger().info("Drawing lidar in frame: {}".format(args.target_frame))

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
                self.args.target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=self.args.tf_timeout),
            )
            points_z1 = transform_points(points_lidar, transform).astype(np.float32)
            points_z1 = downsample_points(points_z1, self.args.max_points)

            with self.lock:
                self.latest_points_z1 = points_z1
                self.last_lidar_time = time.monotonic()
                self.last_status = "ok"
        except Exception as exc:
            self.set_status(str(exc))

    def image_callback(self, msg):
        try:
            rgb = ros_image_to_rgb(msg)
            if self.args.image_width > 0:
                scale = self.args.image_width / float(rgb.shape[1])
                size = (self.args.image_width, max(1, int(round(rgb.shape[0] * scale))))
                rgb = np.asarray(PilImage.fromarray(rgb).resize(size, PilImage.Resampling.BILINEAR))

            with self.lock:
                self.latest_image_rgb = rgb
                self.image_version += 1
                self.last_image_time = time.monotonic()
        except Exception as exc:
            self.set_status(str(exc))

    def snapshot(self):
        with self.lock:
            points = self.latest_points_z1.copy()
            image = None if self.latest_image_rgb is None else self.latest_image_rgb.copy()
            image_version = self.image_version
            status = self.last_status
            lidar_age = time.monotonic() - self.last_lidar_time if self.last_lidar_time else None
            image_age = time.monotonic() - self.last_image_time if self.last_image_time else None
        return points, image, image_version, status, lidar_age, image_age

    def set_status(self, status):
        with self.lock:
            self.last_status = status

        now = self.get_clock().now()
        if self.last_warn_time is None or (now - self.last_warn_time).nanoseconds > 2_000_000_000:
            self.get_logger().warn(status)
            self.last_warn_time = now


class CalibrationWindow(QtWidgets.QMainWindow):
    def __init__(self, args, ros_bridge):
        super().__init__()
        self.args = args
        self.ros_bridge = ros_bridge
        self.last_image_version = -1
        self.vtk_ready = False

        self.setWindowTitle("panolidar angular calibration")
        self.resize(1280, 820)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.vtk_widget = QVTKRenderWindowInteractor(central)
        layout.addWidget(self.vtk_widget, stretch=1)

        self.status_label = QtWidgets.QLabel("waiting for data")
        layout.addWidget(self.status_label)

        controls = QtWidgets.QGridLayout()
        layout.addLayout(controls)

        self.yaw_slider, self.yaw_value = self.add_slider(controls, 0, "yaw", -180, 180, args.yaw, "deg")
        self.pitch_slider, self.pitch_value = self.add_slider(controls, 1, "pitch", -90, 90, args.pitch, "deg")
        self.roll_slider, self.roll_value = self.add_slider(controls, 2, "roll", -180, 180, args.roll, "deg")
        self.patch_yaw_min_slider, self.patch_yaw_min_value = self.add_slider(
            controls, 3, "patch yaw min", -180, 180, args.patch_yaw_min, "deg"
        )
        self.patch_yaw_max_slider, self.patch_yaw_max_value = self.add_slider(
            controls, 4, "patch yaw max", -180, 180, args.patch_yaw_max, "deg"
        )
        self.patch_elevation_min_slider, self.patch_elevation_min_value = self.add_slider(
            controls, 5, "patch elev min", -90, 90, args.patch_elevation_min, "deg"
        )
        self.patch_elevation_max_slider, self.patch_elevation_max_value = self.add_slider(
            controls, 6, "patch elev max", -90, 90, args.patch_elevation_max, "deg"
        )
        self.point_scale_slider, self.point_scale_value = self.add_slider(
            controls, 7, "point cloud scale", 1, 500, int(round(args.point_scale * 100.0)), "scale"
        )
        self.view_pitch_slider, self.view_pitch_lock = self.add_view_axis_control(
            controls, 8, "view pitch", args.view_pitch, args.lock_view_pitch, self.update_camera_view
        )
        self.view_roll_slider, self.view_roll_lock = self.add_view_axis_control(
            controls, 9, "view roll", args.view_roll, args.lock_view_roll, self.update_camera_view
        )
        self.view_yaw_slider, self.view_yaw_lock = self.add_view_axis_control(
            controls, 10, "view yaw", args.view_yaw, args.lock_view_yaw, self.update_camera_view
        )
        self.save_button = QtWidgets.QPushButton("Save calibration")
        self.save_button.clicked.connect(self.save_calibration)
        controls.addWidget(self.save_button, 11, 0, 1, 2)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.02, 0.025, 0.03)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.point_poly_data = vtk.vtkPolyData()
        self.point_mapper = vtk.vtkPolyDataMapper()
        self.point_mapper.SetInputData(self.point_poly_data)
        self.point_mapper.SetScalarModeToUsePointData()
        self.point_mapper.SetColorModeToDirectScalars()
        self.point_mapper.ScalarVisibilityOn()
        self.point_actor = vtk.vtkActor()
        self.point_actor.SetMapper(self.point_mapper)
        self.point_actor.GetProperty().SetPointSize(args.point_size)
        self.renderer.AddActor(self.point_actor)

        self.texture = vtk.vtkTexture()
        self.texture.InterpolateOn()
        self.texture.RepeatOn()
        self.texture.SetInputData(make_default_texture_image())

        self.patch_poly_data = vtk.vtkPolyData()
        self.patch_mapper = vtk.vtkPolyDataMapper()
        self.patch_mapper.SetInputData(self.patch_poly_data)
        self.patch_actor = vtk.vtkActor()
        self.patch_actor.SetMapper(self.patch_mapper)
        self.patch_actor.SetTexture(self.texture)
        self.patch_actor.GetProperty().SetOpacity(args.image_opacity)
        self.patch_actor.GetProperty().LightingOff()
        self.renderer.AddActor(self.patch_actor)

        self.ray_poly_data = vtk.vtkPolyData()
        self.ray_mapper = vtk.vtkPolyDataMapper()
        self.ray_mapper.SetInputData(self.ray_poly_data)
        self.ray_actor = vtk.vtkActor()
        self.ray_actor.SetMapper(self.ray_mapper)
        self.ray_actor.GetProperty().SetColor(1.0, 0.85, 0.05)
        self.ray_actor.GetProperty().SetLineWidth(args.ray_width)
        self.renderer.AddActor(self.ray_actor)

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.5, 1.5, 1.5)
        self.renderer.AddActor(axes)

        self.update_camera_view()
        self.update_patch_geometry()

        for slider in (
            self.yaw_slider,
            self.pitch_slider,
            self.roll_slider,
            self.patch_yaw_min_slider,
            self.patch_yaw_max_slider,
            self.patch_elevation_min_slider,
            self.patch_elevation_max_slider,
        ):
            slider.valueChanged.connect(self.update_patch_geometry)

        self.vtk_widget.Initialize()
        self.vtk_ready = True

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(int(1000.0 / max(1.0, args.update_hz)))

    def add_slider(self, layout, row, label, minimum, maximum, initial, value_kind):
        name = QtWidgets.QLabel(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        value = QtWidgets.QLabel()
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(int(round(initial)))
        slider.setTickInterval(15)
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        value.setMinimumWidth(64)

        def update_label(new_value):
            if value_kind == "scale":
                value.setText("{:.2f}x".format(new_value / 100.0))
            else:
                value.setText("{:+d} deg".format(new_value))

        slider.valueChanged.connect(update_label)
        update_label(slider.value())

        layout.addWidget(name, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value, row, 2)
        return slider, value

    def add_view_axis_control(self, layout, row, label, initial, locked, changed_callback):
        name = QtWidgets.QLabel(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        value = QtWidgets.QLabel()
        lock = QtWidgets.QCheckBox("lock")
        home = QtWidgets.QPushButton("Home")

        slider.setMinimum(-180)
        slider.setMaximum(180)
        slider.setValue(int(round(initial)))
        slider.setTickInterval(15)
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        value.setMinimumWidth(64)

        def update_label(new_value):
            value.setText("{:+d} deg".format(new_value))

        def update_lock(is_locked):
            slider.setEnabled(not is_locked)

        slider.valueChanged.connect(update_label)
        slider.valueChanged.connect(changed_callback)
        lock.toggled.connect(update_lock)
        lock.toggled.connect(changed_callback)
        home.clicked.connect(lambda: slider.setValue(0))
        update_label(slider.value())
        lock.setChecked(locked)
        update_lock(locked)

        layout.addWidget(name, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value, row, 2)
        layout.addWidget(lock, row, 3)
        layout.addWidget(home, row, 4)
        return slider, lock

    def update_camera_view(self, *_args):
        if not hasattr(self, "renderer"):
            return
        for attr in (
            "view_yaw_lock",
            "view_yaw_slider",
            "view_pitch_lock",
            "view_pitch_slider",
            "view_roll_lock",
            "view_roll_slider",
        ):
            if not hasattr(self, attr):
                return

        yaw = 0.0 if self.view_yaw_lock.isChecked() else float(self.view_yaw_slider.value())
        pitch = 0.0 if self.view_pitch_lock.isChecked() else float(self.view_pitch_slider.value())
        roll = 0.0 if self.view_roll_lock.isChecked() else float(self.view_roll_slider.value())

        rotation = euler_zyx_degrees_to_matrix(yaw, pitch, roll)
        base_direction = np.array([1.0, -1.0, 0.7], dtype=np.float32)
        base_direction /= np.linalg.norm(base_direction)
        direction = rotation @ base_direction
        up = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        camera = self.renderer.GetActiveCamera()
        distance = self.args.view_range * 2.0
        camera.SetPosition(*(direction * distance))
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()
        if self.vtk_ready:
            self.vtk_widget.GetRenderWindow().Render()

    def update_scene(self):
        points, image, image_version, status, lidar_age, image_age = self.ros_bridge.snapshot()
        self.update_point_cloud(points)

        if image is not None and image_version != self.last_image_version:
            self.update_texture(image)
            self.last_image_version = image_version

        self.status_label.setText(
            "points: {} | image: {} | lidar age: {} | image age: {} | status: {}".format(
                len(points),
                "yes" if image is not None else "no",
                format_age(lidar_age),
                format_age(image_age),
                status,
            )
        )
        self.vtk_widget.GetRenderWindow().Render()

    def update_point_cloud(self, points):
        colors = np.zeros((0, 3), dtype=np.uint8)
        if len(points):
            colors = make_point_colors(
                points,
                self.current_image_rotation(),
                float(self.patch_yaw_min_slider.value()),
                float(self.patch_yaw_max_slider.value()),
                float(self.patch_elevation_min_slider.value()),
                float(self.patch_elevation_max_slider.value()),
            )
            points = points * (self.point_scale_slider.value() / 100.0)

        vtk_points = vtk.vtkPoints()
        if len(points):
            vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))

        vertices = vtk.vtkCellArray()
        for index in range(len(points)):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(index)

        self.point_poly_data.SetPoints(vtk_points)
        self.point_poly_data.SetVerts(vertices)
        if len(colors):
            vtk_colors = numpy_support.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_colors.SetName("point_colors")
            vtk_colors.SetNumberOfComponents(3)
            self.point_poly_data.GetPointData().SetScalars(vtk_colors)
        self.point_poly_data.Modified()

    def update_texture(self, image_rgb):
        if self.args.flip_vertical:
            image_rgb = np.flipud(image_rgb)
        if self.args.flip_horizontal:
            image_rgb = np.fliplr(image_rgb)

        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        height, width = image_rgb.shape[:2]

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        scalars = numpy_support.numpy_to_vtk(
            image_rgb.reshape(-1, 3),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        vtk_image.GetPointData().SetScalars(scalars)
        vtk_image.Modified()

        self.texture.SetInputData(vtk_image)
        self.texture.Modified()

    def update_patch_geometry(self):
        yaw_min = float(self.patch_yaw_min_slider.value())
        yaw_max = float(self.patch_yaw_max_slider.value())
        elevation_min = float(self.patch_elevation_min_slider.value())
        elevation_max = float(self.patch_elevation_max_slider.value())
        rotation = self.current_image_rotation()
        patch_points, patch_tcoords = make_patch_mesh(
            yaw_min,
            yaw_max,
            elevation_min,
            elevation_max,
            self.args.patch_radius,
            self.args.patch_columns,
            self.args.patch_rows,
            rotation,
        )
        self.set_patch_poly_data(patch_points, patch_tcoords)
        self.set_ray_poly_data(
            make_patch_corner_rays(
                yaw_min,
                yaw_max,
                elevation_min,
                elevation_max,
                self.args.ray_length,
                rotation,
            )
        )
        if self.vtk_ready:
            self.vtk_widget.GetRenderWindow().Render()

    def current_image_rotation(self):
        return euler_zyx_degrees_to_matrix(
            float(self.yaw_slider.value()),
            float(self.pitch_slider.value()),
            float(self.roll_slider.value()),
        )

    def save_calibration(self):
        path = Path(self.args.calibration_path).expanduser().resolve()
        payload = {
            "yaw": float(self.yaw_slider.value()),
            "pitch": float(self.pitch_slider.value()),
            "roll": float(self.roll_slider.value()),
            "patch_yaw_min": float(self.patch_yaw_min_slider.value()),
            "patch_yaw_max": float(self.patch_yaw_max_slider.value()),
            "patch_elevation_min": float(self.patch_elevation_min_slider.value()),
            "patch_elevation_max": float(self.patch_elevation_max_slider.value()),
            "target_frame": self.args.target_frame,
            "source_frame": self.args.source_frame,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.status_label.setText("saved calibration to {}".format(path))

    def set_patch_poly_data(self, points, tcoords):
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(points.reshape(-1, 3), deep=True))

        vtk_tcoords = numpy_support.numpy_to_vtk(tcoords.reshape(-1, 2), deep=True)
        vtk_tcoords.SetName("TextureCoordinates")

        cells = vtk.vtkCellArray()
        rows, columns = points.shape[:2]
        for row in range(rows - 1):
            for column in range(columns - 1):
                p0 = row * columns + column
                p1 = p0 + 1
                p2 = p0 + columns + 1
                p3 = p0 + columns
                quad = vtk.vtkQuad()
                quad.GetPointIds().SetId(0, p0)
                quad.GetPointIds().SetId(1, p1)
                quad.GetPointIds().SetId(2, p2)
                quad.GetPointIds().SetId(3, p3)
                cells.InsertNextCell(quad)

        self.patch_poly_data.SetPoints(vtk_points)
        self.patch_poly_data.SetPolys(cells)
        self.patch_poly_data.GetPointData().SetTCoords(vtk_tcoords)
        self.patch_poly_data.Modified()

    def set_ray_poly_data(self, ray_segments):
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(ray_segments.reshape(-1, 3), deep=True))

        lines = vtk.vtkCellArray()
        for index in range(len(ray_segments)):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, index * 2)
            line.GetPointIds().SetId(1, index * 2 + 1)
            lines.InsertNextCell(line)

        self.ray_poly_data.SetPoints(vtk_points)
        self.ray_poly_data.SetLines(lines)
        self.ray_poly_data.Modified()

    def closeEvent(self, event):
        if rclpy.ok():
            rclpy.shutdown()
        event.accept()


def downsample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def make_default_texture_image():
    image_rgb = np.array(
        [
            [[50, 50, 55], [85, 85, 90]],
            [[85, 85, 90], [50, 50, 55]],
        ],
        dtype=np.uint8,
    )
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(2, 2, 1)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    scalars = numpy_support.numpy_to_vtk(
        image_rgb.reshape(-1, 3),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    vtk_image.GetPointData().SetScalars(scalars)
    vtk_image.Modified()
    return vtk_image


def make_patch_mesh(yaw_min_deg, yaw_max_deg, elevation_min_deg, elevation_max_deg, radius, columns, rows, rotation):
    yaw_min_deg, yaw_max_deg = sorted((yaw_min_deg, yaw_max_deg))
    elevation_min_deg, elevation_max_deg = sorted((elevation_min_deg, elevation_max_deg))
    yaw = np.deg2rad(np.linspace(yaw_min_deg, yaw_max_deg, columns))
    elevation = np.deg2rad(np.linspace(elevation_max_deg, elevation_min_deg, rows))
    yaw_grid, elevation_grid = np.meshgrid(yaw, elevation)

    directions = directions_from_yaw_elevation(yaw_grid, elevation_grid)
    rotated = directions.reshape(-1, 3) @ rotation.T
    points = (rotated.reshape(rows, columns, 3) * radius).astype(np.float32)

    # Texture coordinates sample the centered image patch from the equirectangular image.
    u = (yaw_grid + np.pi) / (2.0 * np.pi)
    v = (np.pi / 2.0 - elevation_grid) / np.pi
    tcoords = np.dstack((u, 1.0 - v)).astype(np.float32)
    return points, tcoords


def make_patch_corner_rays(yaw_min_deg, yaw_max_deg, elevation_min_deg, elevation_max_deg, ray_length, rotation):
    yaw_min_deg, yaw_max_deg = sorted((yaw_min_deg, yaw_max_deg))
    elevation_min_deg, elevation_max_deg = sorted((elevation_min_deg, elevation_max_deg))
    yaw_min = np.deg2rad(yaw_min_deg)
    yaw_max = np.deg2rad(yaw_max_deg)
    elevation_min = np.deg2rad(elevation_min_deg)
    elevation_max = np.deg2rad(elevation_max_deg)
    corners = [
        (yaw_min, elevation_max),
        (yaw_max, elevation_max),
        (yaw_max, elevation_min),
        (yaw_min, elevation_min),
    ]

    segments = []
    for yaw, elevation in corners:
        direction = directions_from_yaw_elevation(np.array(yaw), np.array(elevation)).reshape(3)
        direction = rotation @ direction
        segments.append(np.vstack((np.zeros(3, dtype=np.float32), direction * ray_length)))
    return np.asarray(segments, dtype=np.float32)


def make_point_colors(points_z1, image_rotation, yaw_min_deg, yaw_max_deg, elevation_min_deg, elevation_max_deg):
    colors = np.empty((len(points_z1), 3), dtype=np.uint8)
    colors[:, :] = np.array([0, 210, 255], dtype=np.uint8)
    mask = points_inside_image_patch(points_z1, image_rotation, yaw_min_deg, yaw_max_deg, elevation_min_deg, elevation_max_deg)
    colors[mask] = np.array([0, 255, 80], dtype=np.uint8)
    return colors


def points_inside_image_patch(points_z1, image_rotation, yaw_min_deg, yaw_max_deg, elevation_min_deg, elevation_max_deg):
    if len(points_z1) == 0:
        return np.zeros((0,), dtype=bool)

    yaw_min_deg, yaw_max_deg = sorted((yaw_min_deg, yaw_max_deg))
    elevation_min_deg, elevation_max_deg = sorted((elevation_min_deg, elevation_max_deg))

    norms = np.linalg.norm(points_z1, axis=1)
    valid = norms > 1e-6
    directions_z1 = np.zeros_like(points_z1, dtype=np.float32)
    directions_z1[valid] = points_z1[valid] / norms[valid, None]

    # image_rotation maps camera-image rays into camera_360_link, so transpose maps camera_360_link rays back to image rays.
    directions_camera = directions_z1 @ image_rotation
    yaw = np.rad2deg(np.arctan2(directions_camera[:, 1], directions_camera[:, 0]))
    elevation = np.rad2deg(np.arctan2(directions_camera[:, 2], np.hypot(directions_camera[:, 0], directions_camera[:, 1])))

    return (
        valid
        & (yaw >= yaw_min_deg)
        & (yaw <= yaw_max_deg)
        & (elevation >= elevation_min_deg)
        & (elevation <= elevation_max_deg)
    )


def directions_from_yaw_elevation(yaw, elevation):
    cos_el = np.cos(elevation)
    return np.dstack(
        (
            cos_el * np.cos(yaw),
            cos_el * np.sin(yaw),
            np.sin(elevation),
        )
    )


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


def format_age(age):
    if age is None:
        return "--"
    return "{:.1f}s".format(age)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the lidar cloud in the camera frame with a rotatable 360 camera image patch and corner rays."
    )
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--target-frame", default=DEFAULT_TARGET_FRAME)
    parser.add_argument("--source-frame", default=DEFAULT_SOURCE_FRAME)
    parser.add_argument("--yaw", type=float, default=-1.0)
    parser.add_argument("--pitch", type=float, default=-6.0)
    parser.add_argument("--roll", type=float, default=-180.0)
    parser.add_argument("--view-range", type=float, default=10.0)
    parser.add_argument("--patch-radius", type=float, default=3.0)
    parser.add_argument("--patch-yaw-min", type=int, default=-35)
    parser.add_argument("--patch-yaw-max", type=int, default=35)
    parser.add_argument("--patch-elevation-min", type=int, default=-22)
    parser.add_argument("--patch-elevation-max", type=int, default=22)
    parser.add_argument("--patch-columns", type=int, default=40)
    parser.add_argument("--patch-rows", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=12.0)
    parser.add_argument("--ray-width", type=float, default=3.0)
    parser.add_argument("--image-opacity", type=float, default=1.0)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--point-scale", type=float, default=1.8)
    parser.add_argument("--view-pitch", type=float, default=0.0)
    parser.add_argument("--view-roll", type=float, default=0.0)
    parser.add_argument("--view-yaw", type=float, default=0.0)
    parser.add_argument("--lock-view-pitch", action="store_true")
    parser.add_argument("--lock-view-roll", action="store_true")
    parser.add_argument("--lock-view-yaw", action="store_true", default=True)
    parser.add_argument("--unlock-view-yaw", action="store_false", dest="lock_view_yaw")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--update-hz", type=float, default=10.0)
    parser.add_argument("--queue-depth", type=int, default=10)
    parser.add_argument("--tf-timeout", type=float, default=0.05)
    parser.add_argument("--topic-wait-timeout", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication([])

    rclpy.init()
    ros_bridge = RosDataBridge(args)
    window = CalibrationWindow(args, ros_bridge)
    window.show()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_bridge,), daemon=True)
    spin_thread.start()

    try:
        app.exec_()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        ros_bridge.destroy_node()


if __name__ == "__main__":
    main()

# panolidar

`panolidar` is a ROS 2 toolkit for calibrating a 360 camera to a lidar, selecting lidar points from a 360 image bounding box, and querying bbox-based depth, position, and bearing.

The package is sensor-agnostic. It was developed with a stitched equirectangular 360 camera image and a 3D lidar cloud, but the code only assumes:

- the camera image is equirectangular
- the lidar message provides XYZ points
- ROS TF can transform lidar points into the 360 camera frame

## Files

```text
scripts/spherical_lidar_camera.py
  Shared point cloud decoding, TF transform, and spherical filtering helpers.

scripts/panolidar_calibration_viewer.py
  3D calibration GUI. Shows lidar points, a selectable 360 image patch, corner rays,
  and saves yaw/pitch/roll calibration.

scripts/panolidar_bbox_point_selector.py
  Image GUI. Drag a bbox on the 360 image, select matching lidar points, and
  publish them as PointCloud2.

scripts/panolidar_bbox_inference_node.py
  ROS 2 service node. Receives image bboxes and returns depth, 3D position, or
  XY distance/angle using the latest lidar cloud.

scripts/panolidar_bbox_service_tester.py
  Image GUI for testing the inference services. Drag a bbox, choose the service,
  aggregation mode, outlier filter, and inspect result latency.

srv/PanoBBoxDepth.srv
srv/PanoBBoxPosition.srv
srv/PanoBBoxBearing.srv
  Service definitions used by the bbox inference node.

config/panolidar_calibration_params.json
  Saved/default calibration parameters.

urdf/panolidar_frames.urdf.xacro
  URDF snippet for generic 360 lidar and 360 camera frames.
```

## Dependencies

```bash
sudo apt install \
  ros-$ROS_DISTRO-rclpy \
  ros-$ROS_DISTRO-tf2-ros \
  ros-$ROS_DISTRO-geometry-msgs \
  ros-$ROS_DISTRO-sensor-msgs-py \
  ros-$ROS_DISTRO-visualization-msgs \
  ros-$ROS_DISTRO-rosidl-default-generators \
  python3-pyqt5 \
  python3-vtk9 \
  python3-pil
```

Your lidar driver must publish either `sensor_msgs/msg/PointCloud2` or a custom message with `points[].x/y/z`.

## Frames

The current defaults match the original development robot:

```text
lidar topic:         /livox/lidar
lidar source frame:  livox_link
camera target frame: z1_link
robot output frame:  base_link
image topic:         /image_raw
```

For a more generic robot, use names like:

```text
lidar source frame:  lidar_360_link
camera target frame: camera_360_link
robot output frame:  base_link
```

Add fixed links for the lidar and 360 camera. Replace the poses with your measured mounting geometry:

```xml
<link name="lidar_360_link">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.05" radius="0.03"/>
    </geometry>
  </visual>
</link>

<joint name="lidar_360_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_360_link"/>
  <origin xyz="0.174 0.0 0.153" rpy="0 0 0"/>
</joint>

<link name="camera_360_link">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.05" radius="0.03"/>
    </geometry>
  </visual>
</link>

<joint name="camera_360_joint" type="fixed">
  <parent link="lidar_360_link"/>
  <child link="camera_360_link"/>
  <origin xyz="-0.04 0.05 0.30" rpy="0 0 0"/>
</joint>
```

Verify TF:

```bash
ros2 run tf2_ros tf2_echo camera_360_link lidar_360_link
```

If your hardware uses the generic names, pass them through the command-line arguments or update the defaults in the scripts.

## Build

The inference services use custom service definitions, so build the package with `colcon`:

```bash
cd ~/Playground
colcon build --packages-select panolidar --symlink-install
source install/setup.bash
```

Check the install:

```bash
ros2 pkg executables panolidar
ros2 interface show panolidar/srv/PanoBBoxBearing
```

## Calibrate 360 Image Rotation

Run:

```bash
cd ~/Playground/panolidar
python3 scripts/panolidar_calibration_viewer.py \
  --lidar-topic /livox/lidar \
  --image-topic /image_raw \
  --target-frame z1_link \
  --source-frame livox_link \
  --calibration-path config/panolidar_calibration_params.json
```

The calibration file stores yaw, pitch, and roll for the stitched 360 image orientation relative to the camera frame, `z1_link` by default.

```json
{
  "pitch": -6.0,
  "roll": -180.0,
  "yaw": -1.0
}
```

Keep the TF transform and image rotation conceptually separate:

```text
URDF/TF: where the sensors are mounted
calibration JSON: how the stitched 360 image's spherical coordinates line up with the camera frame
```

## Select BBox Points

Run:

```bash
python3 scripts/panolidar_bbox_point_selector.py \
  --lidar-topic /livox/lidar \
  --image-topic /image_raw \
  --output-topic /vis_selected_points \
  --target-frame z1_link \
  --source-frame livox_link \
  --calibration-path config/panolidar_calibration_params.json
```

Drag a rectangle on the 360 image. The script:

1. Converts the pixel bbox to equirectangular yaw/elevation bounds.
2. Applies the saved image rotation calibration.
3. Transforms incoming lidar points into the camera frame, `z1_link` by default.
4. Selects points whose direction falls inside the bbox.
5. Publishes selected points to `/vis_selected_points`.

In RViz, add a `PointCloud2` display:

```text
Topic: /vis_selected_points
Fixed frame: z1_link
```

## Run BBox Inference Services

Start the service node:

```bash
source ~/Playground/install/setup.bash
ros2 run panolidar panolidar_bbox_inference_node.py \
  --lidar-topic /livox/lidar \
  --source-frame livox_link \
  --camera-frame z1_link \
  --default-frame base_link \
  --calibration-path ~/Playground/panolidar/config/panolidar_calibration_params.json
```

Services:

```text
/bbox_depth     -> representative Euclidean depth
/bbox_position  -> representative PointStamped and depth
/bbox_bearing   -> representative PointStamped, depth, XY distance, and yaw angle
```

Each service request contains:

```text
x_min, y_min, x_max, y_max  pixel bbox in the 360 image
image_width, image_height   image dimensions used for bbox conversion
frame_id                    output frame; empty uses --default-frame
method                      center_k, all, or main_cluster
k                           number of center-ray points for center_k
outlier_filter              none, mad_range, statistical, or radius
```

The result is computed in the requested output frame. With the default `base_link`, `xy_distance_m` is `sqrt(x*x + y*y)` and `angle_rad/angle_deg` is `atan2(y, x)`.

## Test Services With GUI

Start the GUI tester:

```bash
source ~/Playground/install/setup.bash
ros2 run panolidar panolidar_bbox_service_tester.py \
  --image-topic /image_raw
```

Use the tester to:

1. Draw a bbox on the live 360 image.
2. Choose output frame, aggregation method, `k`, and outlier filter.
3. Press `Depth`, `Position`, or `Bearing`.
4. Read the result, point counts, and round-trip service time in milliseconds.

## RViz Visualization

Every service call publishes:

```text
/vis_bbox_selected_points  all lidar points inside the bbox before outlier filtering
/vis_bbox_used_points      exact points used to compute the final average
/vis_bbox_outlier_points   points rejected by the selected outlier filter
/vis_bbox_result_marker    orange sphere marker at the calculated representative point
```

Add these RViz displays:

```text
PointCloud2  /vis_bbox_selected_points
PointCloud2  /vis_bbox_used_points
PointCloud2  /vis_bbox_outlier_points
Marker       /vis_bbox_result_marker
```

Set RViz `Fixed Frame` to the service output frame, usually `base_link`.

You can rename the topics:

```bash
ros2 run panolidar panolidar_bbox_inference_node.py \
  --selected-points-topic /vis_bbox_selected_points \
  --used-points-topic /vis_bbox_used_points \
  --outlier-points-topic /vis_bbox_outlier_points \
  --result-marker-topic /vis_bbox_result_marker \
  --marker-scale 0.18
```

## Aggregation Options

```text
center_k
  Sorts candidate points by angular closeness to the bbox center ray and averages
  only the best k points. /vis_bbox_used_points contains exactly those k points.

all
  Averages every point that remains after outlier filtering.

main_cluster
  Clusters the filtered points in 3D and averages the largest cluster.
```

## Outlier Options

```text
none
  Keep every selected point.

mad_range
  Median Absolute Deviation filtering on range. It rejects points whose distance
  from the median range is too large. This is robust to a few very near/far points
  and is the default.

statistical
  Computes each point's average distance to its nearest neighbors and rejects
  points that are much more isolated than the group.

radius
  Keeps a point only if it has enough neighbors within a fixed 3D radius.
```

Good starting combinations:

```text
center_k + mad_range
main_cluster + mad_range
main_cluster + radius
```

RANSAC can be useful later if you want to remove a known geometric model, such as a ground plane or background wall, but it is not the default because generic object bboxes do not always contain a single plane, line, cylinder, or sphere.

## Coordinate Model

The 360 image is treated as equirectangular:

```text
horizontal pixel u -> yaw
vertical pixel v   -> elevation
```

The current bbox selector uses the vertical convention observed in this setup:

```text
top of image    -> lower elevation
bottom of image -> higher elevation
```

If your stream is vertically inverted, swap the elevation mapping in `pixel_bbox_to_angle_bounds()`.

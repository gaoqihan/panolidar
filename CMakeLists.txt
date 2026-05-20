cmake_minimum_required(VERSION 3.8)
project(panolidar)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PanoBBoxDepth.srv"
  "srv/PanoBBoxPosition.srv"
  "srv/PanoBBoxBearing.srv"
  DEPENDENCIES geometry_msgs
)

install(PROGRAMS
  scripts/spherical_lidar_camera.py
  scripts/panolidar_calibration_viewer.py
  scripts/panolidar_bbox_inference_node.py
  scripts/panolidar_bbox_service_tester.py
  scripts/panolidar_bbox_point_selector.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

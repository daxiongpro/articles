cmake_minimum_required(VERSION 3.5)
project(pointcloud)

# 找到ROS2依赖项并添加头文件路径
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(main main.cpp)

# 添加头文件路径
include_directories(
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
)

# 链接ROS2库
ament_target_dependencies(main rclcpp sensor_msgs)


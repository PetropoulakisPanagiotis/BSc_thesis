cmake_minimum_required(VERSION 2.8.3)
project(visual_servo)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_filters
  nav_msgs
  geometry_msgs
  cv_bridge
  kinect2_bridge
  message_runtime
)

find_package(OpenCV REQUIRED)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kinect_handler
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# Petropoulakis Panagiotis

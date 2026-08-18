cmake_minimum_required(VERSION 3.0.2)
project(dep_car_evaluation)
find_package(catkin REQUIRED COMPONENTS rospy gazebo_msgs geometry_msgs nav_msgs std_msgs dep_car_core dep_car_msgs)
catkin_package(CATKIN_DEPENDS rospy gazebo_msgs geometry_msgs nav_msgs std_msgs dep_car_core dep_car_msgs)
catkin_install_python(PROGRAMS
  scripts/evaluation_node.py
  scripts/run_p6_static_episode.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

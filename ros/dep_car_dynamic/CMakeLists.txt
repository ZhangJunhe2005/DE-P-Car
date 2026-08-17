cmake_minimum_required(VERSION 3.0.2)
project(dep_car_dynamic)
find_package(catkin REQUIRED COMPONENTS rospy geometry_msgs dep_car_core dep_car_msgs)
catkin_package(CATKIN_DEPENDS rospy geometry_msgs dep_car_core dep_car_msgs)
catkin_install_python(PROGRAMS scripts/dynamic_tracker_node.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


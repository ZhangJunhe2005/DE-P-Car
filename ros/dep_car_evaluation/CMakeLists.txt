cmake_minimum_required(VERSION 3.0.2)
project(dep_car_evaluation)
find_package(catkin REQUIRED COMPONENTS rospy gazebo_msgs nav_msgs dep_car_msgs)
catkin_package(CATKIN_DEPENDS rospy gazebo_msgs nav_msgs dep_car_msgs)
catkin_install_python(PROGRAMS scripts/evaluation_node.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


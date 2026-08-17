cmake_minimum_required(VERSION 3.0.2)
project(dep_car_runtime_safety)
find_package(catkin REQUIRED COMPONENTS dep_car_core dep_car_msgs)
catkin_package(CATKIN_DEPENDS dep_car_core dep_car_msgs)


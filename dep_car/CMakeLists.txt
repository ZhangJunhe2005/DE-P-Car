cmake_minimum_required(VERSION 3.0.2)
project(dep_car_core)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


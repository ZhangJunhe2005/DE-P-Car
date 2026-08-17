cmake_minimum_required(VERSION 3.0.2)
project(dep_car_bringup)
find_package(catkin REQUIRED)
catkin_package()
install(PROGRAMS
  scripts/gzserver_exec.sh
  scripts/gzclient_exec.sh
  scripts/rviz_exec.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

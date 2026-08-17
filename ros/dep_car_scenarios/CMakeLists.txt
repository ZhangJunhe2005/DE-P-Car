cmake_minimum_required(VERSION 3.0.2)
project(dep_car_scenarios)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS scripts/generate_dynamic_scenarios.py scripts/build_actor_world.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY scenarios DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


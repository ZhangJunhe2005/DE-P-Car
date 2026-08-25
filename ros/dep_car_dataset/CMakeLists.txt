cmake_minimum_required(VERSION 3.0.2)
project(dep_car_dataset)
find_package(catkin REQUIRED COMPONENTS rospy message_filters tf2_ros sensor_msgs nav_msgs geometry_msgs gazebo_msgs std_srvs dep_car_core dep_car_msgs)
catkin_package(CATKIN_DEPENDS rospy message_filters tf2_ros sensor_msgs nav_msgs geometry_msgs gazebo_msgs std_srvs dep_car_core dep_car_msgs)
catkin_install_python(PROGRAMS scripts/generate_static_maps.py scripts/generate_static_dataset.py scripts/generate_pilot_tasks.py scripts/run_pilot_episode.py scripts/reset_pilot_pose.py scripts/run_pilot_collection.py scripts/reextract_pilot_samples.py scripts/collect_static_sample.py scripts/collect_multimodal_sample.py scripts/extract_multimodal_bag.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(PROGRAMS
  scripts/record_multimodal_episode.sh
  scripts/record_v43_dagger_episode.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

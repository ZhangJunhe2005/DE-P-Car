cmake_minimum_required(VERSION 3.0.2)
project(dep_car_msgs)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs geometry_msgs nav_msgs)
add_message_files(FILES
  AckermannCommand.msg
  Candidate.msg
  CandidateArray.msg
  DynamicTrack.msg
  DynamicTrackArray.msg
  PlannerState.msg
  RoutePoint.msg
  AckermannRoute.msg
  LocalRouteCommand.msg
  PolicyCandidate.msg
  PolicyCandidateArray.msg
  PolicyQuery.msg
  PolicyState.msg
)
generate_messages(DEPENDENCIES std_msgs geometry_msgs nav_msgs)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs nav_msgs)

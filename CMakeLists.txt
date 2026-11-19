cmake_minimum_required(VERSION 3.0.2)
project(roban_person_follower)

find_package(catkin REQUIRED COMPONENTS message_generation sensor_msgs std_msgs)

catkin_python_setup()

add_message_files(FILES Target.msg Status.msg)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime sensor_msgs std_msgs)

catkin_install_python(PROGRAMS scripts/follower_node.py scripts/download_model.py scripts/summarize_run.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES src/roban_person_follower/bytetrack/LICENSE
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} RENAME LICENSE.ByteTrack)

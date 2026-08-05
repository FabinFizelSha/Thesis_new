cmake_minimum_required(VERSION 3.8)
project(rsg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(hydra_msgs REQUIRED)
find_package(spark_dsg REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

# ``nodes`` is the Python package containing the two Python ROS nodes and
# their internal helper modules. The C++ fuser is compiled below as the third node.
ament_python_install_package(nodes)

add_executable(rsg_scene_graph_fuser
  nodes/fuser.cpp
)
target_compile_features(rsg_scene_graph_fuser PUBLIC cxx_std_17)
ament_target_dependencies(rsg_scene_graph_fuser
  rclcpp
  std_msgs
  builtin_interfaces
  geometry_msgs
  visualization_msgs
  hydra_msgs
)
target_link_libraries(rsg_scene_graph_fuser
  spark_dsg::spark_dsg
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RsgFrame.msg"
  "msg/Phase1ClassificationResult.msg"
  "msg/Phase1VlmResult.msg"
  "msg/RsgHydraFrame.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

install(TARGETS rsg_scene_graph_fuser
  DESTINATION lib/${PROJECT_NAME}
)

set(RSG_PYTHON_EXECUTABLES
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/rsg_preprocessor"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/rsg_phase1_semantic_coordinator"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/rsg_play_uhumans2"
)

install(PROGRAMS ${RSG_PYTHON_EXECUTABLES}
  DESTINATION lib/${PROJECT_NAME}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(DIRECTORY
  scripts
  launch
  config
  nodes
  docs
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  README.md
  RSG_PACKAGE_REVISION.txt
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

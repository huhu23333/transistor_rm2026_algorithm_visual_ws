cmake_minimum_required(VERSION 3.5)
project(auto_aim)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# 设置 libtorch 路径
set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")



# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SerialData.msg"
  "msg/GimbalCommand.msg"
)

set(CMAKE_INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")


# 查找 MvCameraControl 库
find_library(MVCAMERA_LIB MvCameraControl PATHS /opt/MVS/lib/64)

# 设置包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    /opt/MVS/include
    /opt/ros/humble/include
    /opt/ros/humble/include/cv_bridge
    ${sensor_msgs_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
)

# 添加装甲板检测相关的库
add_library(armor_detector SHARED
    src/armor_detector/ArmorDetector.cpp
    src/armor_detector/LightBarDetector.cpp
    src/armor_detector/BallisticSolver.cpp
    src/armor_detector/ArmorClassifier.cpp
    src/armor_detector/ArmorAngleKalman.cpp
    src/camera/cameraConnect.cpp  # 添加这一行
    src/test_codes/FrameRateCounter.cpp
)
add_library(crc_lib SHARED
    src/nodes/CRC.cpp
)
# 添加可执行文件
add_executable(armor_detect_node src/nodes/ArmorDetect_Node.cpp)
add_executable(com_node src/nodes/Com_Node.cpp)

# 为装甲板检测器添加依赖
target_link_libraries(armor_detector
    ${OpenCV_LIBS}
    Eigen3::Eigen
    rclcpp::rclcpp   # 添加这一行
)

# 为装甲板检测节点添加依赖
target_link_libraries(armor_detect_node
    armor_detector
    ${OpenCV_LIBS}
    ${MVCAMERA_LIB}
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    pthread
    rclcpp::rclcpp
    ${sensor_msgs_LIBRARIES}
    ${cv_bridge_LIBRARIES}
)

# 为通信节点添加依赖
target_link_libraries(com_node
    crc_lib
    pthread
    rclcpp::rclcpp
)

# 添加ROS2消息依赖
rosidl_target_interfaces(armor_detect_node
    ${PROJECT_NAME} "rosidl_typesupport_cpp")

rosidl_target_interfaces(com_node
    ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -flto -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS "-flto=auto")

# 添加库的路径
link_directories(/opt/MVS/lib/64)

# 安装
install(TARGETS
    armor_detect_node
    com_node
    armor_detector  # 添加这行
    crc_lib        # 添加这行
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
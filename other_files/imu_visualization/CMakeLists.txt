cmake_minimum_required(VERSION 3.5)
project(imu_visualization)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# 设置包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 添加可执行文件
add_executable(imu_visualization imu_visualization.cpp)

# 链接库
target_link_libraries(imu_visualization
    ${OpenCV_LIBS}
    pthread
    udev
)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

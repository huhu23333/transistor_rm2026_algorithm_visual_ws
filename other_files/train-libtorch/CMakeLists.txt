cmake_minimum_required(VERSION 3.5)
project(train-rm2026)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# 设置 libtorch 路径
set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# 添加可执行文件
add_executable(train src/train.cpp)

# 包含头文件目录
target_include_directories(train PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# 链接库目录
target_link_directories(train PRIVATE
    ${LIBTORCH_PATH}/lib
    ${OPENCV_PATH}/lib
)

# 链接库
target_link_libraries(train PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    torch
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
)

# 设置运行时库路径 (RPATH)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  # 自动包含链接路径
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")  # macOS 相对路径
else()
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")      # Linux 相对路径
endif()

# 编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -flto -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS "-flto=auto")

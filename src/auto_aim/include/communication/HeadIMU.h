#ifndef HEAD_IMU_H
#define HEAD_IMU_H

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <vector>
#include <queue>
#include <array>
#include <mutex>
#include <atomic>
#include <chrono>
#include <dirent.h>  // 用于遍历/dev目录
#include <sys/types.h>
#include <sys/stat.h>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include <functional>

#include <iostream>
#include <libudev.h>
#include <thread>


#include <cstring>
#include <sys/ioctl.h>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <memory>
#include <thread>

struct HeadIMUSerialData {
    int32_t rectified_avx_multiply_8;
    int32_t rectified_avy_multiply_8;
    int32_t rectified_avz_multiply_8;
    short received_ax;
    short received_ay;
    short received_az;
    float euler_yaw;
    float euler_pitch;
    float euler_roll;
};

class HeadIMUSerialCommunicationClass {
public:
    HeadIMUSerialCommunicationClass(std::function<void(const HeadIMUSerialData&)> serialDataCallback);
    ~HeadIMUSerialCommunicationClass();
    void timerCallback();
    void timerThread();
    
private:
    struct DataFrame {
        uint8_t header1;
        uint8_t header2;
        uint8_t header3;
        uint8_t data_len;
        int32_t rectified_avx_multiply_8;
        int32_t rectified_avy_multiply_8;
        int32_t rectified_avz_multiply_8;
        short received_ax;
        short received_ay;
        short received_az;
        float euler_yaw;
        float euler_pitch;
        float euler_roll;
        uint32_t crc32;
    };
    static constexpr size_t BUFFER_SIZE = 1024;
    static constexpr uint8_t FRAME_HEADER1 = 0xA7;
    static constexpr uint8_t FRAME_HEADER2 = 0xB6;
    static constexpr uint8_t FRAME_HEADER3 = 0xC5;
    static constexpr size_t FRAME_MIN_SIZE = 38;

    int fd_;
    std::array<uint8_t, BUFFER_SIZE> buffer_;
    size_t buffer_index_ = 0;

    std::function<void(const HeadIMUSerialData&)> serialDataCallback;
    bool running = true;

    std::chrono::steady_clock::time_point last_reconnect_time;
    std::chrono::steady_clock::time_point last_received_time;
    
    void initializeSerial();
    std::vector<std::string> findAvailableSerialPorts();
    void processFrame(const uint8_t* data);
    void processBuffer();
    void tryReconnect();
    std::string getSerialProductInfo(const std::string& port);
};

uint32_t HAL_CRC_Calculate(const uint8_t* data, size_t length);




#endif
// Com.h
#ifndef COM_H
#define COM_H

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
#include "utils/CRC.h"
#include <dirent.h>  // 用于遍历/dev目录
#include <sys/types.h>
#include <sys/stat.h>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <functional>


struct SerialData {
    float bullet_velocity;  // 子弹速度
    float bullet_angle;    // 子弹角度
    int16_t gimbal_yaw;       // 云台当前偏航角
    uint8_t color;            // 敌方颜色(0:红色, 1:蓝色)
};

class SerialCommunicationClass {
public:
    SerialCommunicationClass(rclcpp::Node* node, std::function<void(const SerialData&)> serialDataCallback);
    ~SerialCommunicationClass();
    void timerCallback();
    bool sendData(float pitch_target, float yaw_target);
    void timerThread();
    
private:
    struct DataFrame {
        float bullet_velocity;
        float bullet_angle;
        int16_t gimbal_yaw;
        uint16_t mark;
        uint8_t color;
        float z_rotation_velocity;
    };
    static constexpr size_t BUFFER_SIZE = 1024;
    static constexpr uint8_t FRAME_HEADER1 = 0x42;
    static constexpr uint8_t FRAME_HEADER2 = 0x52;
    static constexpr uint8_t COMMAND_CODE = 0xCD;
    static constexpr size_t FRAME_MIN_SIZE = 5;
    std::mutex queue_mutex_;
    static constexpr size_t MAX_QUEUE_SIZE = 1;

    int fd_;
    std::array<uint8_t, BUFFER_SIZE> buffer_;
    size_t buffer_index_ = 0;
    std::atomic<int> received_commands_count_{0};
    std::atomic<int> sent_commands_count_{0};

    rclcpp::Node* node;
    std::function<void(const SerialData&)> serialDataCallback;
    int error_print_slower = 0;
    bool running = true;
    
    void initializeSerial();
    std::string findAvailableSerialPort();
    void processFrame(const uint8_t* data);
    void processBuffer();
};

#endif // COM_H
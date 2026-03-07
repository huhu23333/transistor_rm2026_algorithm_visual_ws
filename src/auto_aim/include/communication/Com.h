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
#include "communication/CRC.h"
#include <dirent.h>  // 用于遍历/dev目录
#include <sys/types.h>
#include <sys/stat.h>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <functional>
#include <algorithm>
#include <iostream>
#include <libudev.h>
#include <thread>


struct SerialData {
    float bullet_velocity;  // 子弹速度
    float gimbal_pitch;    // 子弹角度
    float gimbal_yaw_small;       // 云台当前偏航角
    float gimbal_yaw_big;       // 云台当前偏航角
    uint8_t color;            // 敌方颜色(0:红色, 1:蓝色)

    std::vector<uint8_t> all_car_rebirth_infos; // 用于判断是否在无敌时间内的变量
};

class SerialCommunicationClass {
public:
    SerialCommunicationClass(rclcpp::Node* node, std::function<void(const SerialData&)> serialDataCallback);
    ~SerialCommunicationClass();
    void timerCallback();
    bool sendData(bool reset, float pitch_target, float small_yaw_target, float big_yaw_target, bool fire, float big_yaw_enemy_position_x, float big_yaw_enemy_position_y);
    void timerThread();
    
private:
    struct DataFrame {
        float bullet_velocity;
        uint16_t gimbal_pitch;
        uint16_t gimbal_yaw_small; // 小yaw相对大yaw角度
        uint16_t mark;
        uint8_t color;
        int16_t z_rotation_velocity;
        // 哨兵新增
        uint8_t chassis_mode; // 底盘模式
        uint8_t lack_blood_son_mode; // 缺血回城子模式
        int16_t gimbal_yaw_big; // 大yaw当前角度(世界坐标系下)，单位rad

        uint8_t all_car_rebirth_infos[4];
    };
    static constexpr size_t BUFFER_SIZE = 1024;
    static constexpr uint8_t FRAME_HEADER1 = 0x42;
    static constexpr uint8_t FRAME_HEADER2 = 0x52;
    static constexpr uint8_t COMMAND_CODE = 0xCD;
    static constexpr size_t FRAME_MIN_SIZE = 5;
    // std::mutex queue_mutex_;
    // static constexpr size_t MAX_QUEUE_SIZE = 1;

    int fd_;
    std::array<uint8_t, BUFFER_SIZE> buffer_;
    size_t buffer_index_ = 0;
    // std::atomic<int> received_commands_count_{0};
    // std::atomic<int> sent_commands_count_{0};

    rclcpp::Node* node;
    std::function<void(const SerialData&)> serialDataCallback;
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

#endif // COM_H2
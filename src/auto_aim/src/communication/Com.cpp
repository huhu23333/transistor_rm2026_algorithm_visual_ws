// Com.cpp
#include "communication/Com.h"

std::string SerialCommunicationClass::getSerialProductInfo(const std::string& port) {
    struct udev *udev;
    struct udev_device *dev;
    std::string result = "";
    
    // 创建udev对象
    udev = udev_new();
    if (!udev) {
        return "Failed to create udev";
    }
    
    // 根据设备路径获取设备信息
    dev = udev_device_new_from_subsystem_sysname(udev, "tty", port.c_str());
    if (!dev) {
        udev_unref(udev);
        return "Device not found";
    }
    
    // 获取父设备（USB设备）
    struct udev_device *parent = udev_device_get_parent_with_subsystem_devtype(
        dev, "usb", "usb_device");
    
    if (parent) {
        // 获取产品信息
        const char *product = udev_device_get_sysattr_value(parent, "product");
        
        if (product) {
            result += std::string(product);
        }
    }
    
    udev_device_unref(dev);
    udev_unref(udev);
    return result;
}

SerialCommunicationClass::SerialCommunicationClass(rclcpp::Node* node, std::function<void(const SerialData&)> serialDataCallback) 
: node(node), serialDataCallback(serialDataCallback), fd_(-1) {
    initializeSerial();
    last_reconnect_time = std::chrono::steady_clock::now();
    last_received_time = std::chrono::steady_clock::now();
}

SerialCommunicationClass::~SerialCommunicationClass() {
    running = false;
    if (fd_ >= 0) {
        close(fd_);
    }
}

void SerialCommunicationClass::tryReconnect() {
    if (fd_ >= 0) {
        close(fd_);
    }
    buffer_index_ = 0;
    initializeSerial();
    last_reconnect_time = std::chrono::steady_clock::now();
    last_received_time = std::chrono::steady_clock::now();
}
    
void SerialCommunicationClass::initializeSerial() {
    std::vector<std::string> ports = findAvailableSerialPorts();
    if (ports.empty()) {
        printf("No available serial port found!\n");
        return;
    }
    std::string port;
    for (auto test_port : ports) {
        try {
            if(getSerialProductInfo(test_port.substr(5)) != std::string("STM32 Virtual ComPort MyIMU")) {
                port = test_port;
                break;
            };
        } catch (...) {

        }
    }
    if (port.empty()) {
        printf("Target serial port Not found!\n");
        return;
    }

    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd_ < 0) {
        RCLCPP_ERROR(node->get_logger(), "Failed to open port %s: %s", port.c_str(), strerror(errno));
        return;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));

    if (tcgetattr(fd_, &tty) != 0) {
        RCLCPP_ERROR(node->get_logger(), "Failed to get serial attributes");
        close(fd_);
        fd_ = -1;
        return;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ISIG;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        RCLCPP_ERROR(node->get_logger(), "Failed to set serial attributes");
        close(fd_);
        fd_ = -1;
        return;
    }

    tcflush(fd_, TCIOFLUSH);
    RCLCPP_DEBUG(node->get_logger(), "Serial initialized: %s", port.c_str());
}

// 查找可用的串口
std::vector<std::string> SerialCommunicationClass::findAvailableSerialPorts() {
    struct dirent *entry;
    DIR *dp = opendir("/dev/");
    if (dp == nullptr) {
        printf("Failed to open /dev/ directory\n");
        return std::vector<std::string>(0);
    }

    std::vector<std::string> ports;
    while ((entry = readdir(dp)) != nullptr) {
        if (strncmp(entry->d_name, "ttyACM", 6) == 0) {  // 匹配ttyACM串口
            std::string candidate_port = "/dev/" + std::string(entry->d_name);
            int fd = open(candidate_port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
            if (fd >= 0) {
                close(fd);  // 串口可用，返回串口名称
                ports.push_back(candidate_port);
                // break;
            }
        }
    }

    closedir(dp);
    return ports;
}

bool SerialCommunicationClass::sendData(bool reset, float pitch_target, float small_yaw_target, float big_yaw_target, bool fire, float big_yaw_enemy_position_x, float big_yaw_enemy_position_y) {
    if (fd_ >= 0) {
        //pitch_target = -0.01; // 约 0.01对应30°
        //small_yaw_target = 0;
        // 传入参数使用弧度制 [-M_PI, M_PI]
        // 总大小 = 帧头(2) + 命令码(1) + 长度(1) + reset(1) + pitch_target(2) + small_yaw_target(2) + big_yaw_enemy_position_x(2) + big_yaw_enemy_position_y(2) + fire(1) + CRC(1) + big_yaw_target(2) = 17字节
        std::array<uint8_t, 17> tx_data{};
        
        tx_data[0] = FRAME_HEADER1;
        tx_data[1] = FRAME_HEADER2;
        tx_data[2] = COMMAND_CODE;
        tx_data[3] = 12;  // 数据长度为12（1字节reset + 2字节pitch_target + 2字节small_yaw_target + 2字节big_yaw_enemy_position_x + 2字节big_yaw_enemy_position_y + 1字节fire + 2字节big_yaw_target）

        // 处理reset
        if (reset) {
            tx_data[4] = 0x01;
        } else {
            tx_data[4] = 0x00;
        }
        
        // 处理pitch_target (2字节)
        pitch_target = pitch_target - 0.40;
        int16_t pitch_int16 = static_cast<int16_t>(pitch_target * (180.0f / M_PI * 173.5f / 60.0f)) + 32844;
        pitch_int16 -= 72; // 机械零点偏移约为88
        uint16_t pitch_uint16;
        if (pitch_int16 >= 0) {
            pitch_uint16 = pitch_int16;
        } else {
            pitch_uint16 = 65536 + pitch_int16;
        }
        memcpy(&tx_data[5], &pitch_uint16, sizeof(uint16_t));  // 2字节
        
        // 处理small_yaw_target (2字节)
        small_yaw_target = std::atan2(std::sin(small_yaw_target), std::cos(small_yaw_target));
        int16_t small_yaw_int16 = static_cast<int16_t>(small_yaw_target * 4096 / M_PI) + 1350;  // 将float转换为定点数
        uint16_t small_yaw_uint16;
        if (small_yaw_int16 >= 0) {
            small_yaw_uint16 = small_yaw_int16;
        } else {
            small_yaw_uint16 = 8192 + small_yaw_int16;
        }
        memcpy(&tx_data[7], &small_yaw_uint16, sizeof(uint16_t));  // 2字节

        // 处理fire
        if (fire) {
            tx_data[9] = 0x01;
        } else {
            tx_data[9] = 0x00;
        }

        // 处理enemy_position (2+2字节)
        int16_t big_yaw_enemy_position_x_int16 = static_cast<int16_t>(std::clamp(big_yaw_enemy_position_x, -32766.0f, 32766.0f));
        int16_t big_yaw_enemy_position_y_int16 = static_cast<int16_t>(std::clamp(big_yaw_enemy_position_y, -32766.0f, 32766.0f));
        memcpy(&tx_data[10], &big_yaw_enemy_position_x_int16, sizeof(int16_t));
        memcpy(&tx_data[12], &big_yaw_enemy_position_y_int16, sizeof(int16_t));
        
        // 处理big_yaw_target (2字节)
        big_yaw_target = -std::atan2(std::sin(big_yaw_target), std::cos(big_yaw_target));
        int16_t big_yaw_int16 = static_cast<int16_t>(big_yaw_target * 4096 / M_PI);
        memcpy(&tx_data[14], &big_yaw_int16, sizeof(int16_t));


        // 计算并添加CRC
        tx_data[16] = CRC8_Check_Sum(tx_data.data(), 16);


        RCLCPP_INFO(node->get_logger(), 
            "\033[1;34m[Send Data]\033[0m\n"
            "\033[1;32mReset:\033[0m %d\n"
            "\033[1;32mPitch_uint16 Angle:\033[0m %d\n"
            "\033[1;33msmall_yaw_uint16:\033[0m %d\n"
            "\033[1;33mbig_yaw_int16:\033[0m %u\n"
            "\033[1;36mFire:\033[0m %d\n",
            reset,
            pitch_uint16,
            small_yaw_uint16,
            big_yaw_int16,
            fire
        );

        ssize_t written = write(fd_, tx_data.data(), tx_data.size());
        if (written == static_cast<ssize_t>(tx_data.size())) {
            RCLCPP_INFO(node->get_logger(), "TX: pitch_target=%.2f small_yaw_target=%.2f(uint16=%u) big_yaw_target=%.2f(uint16=%d)", 
                        pitch_target, small_yaw_target, small_yaw_uint16, big_yaw_target, big_yaw_int16);
            return true;
        } else {
            RCLCPP_INFO(node->get_logger(), "TX write failed: written %ld bytes", 
                        written);
        }
    }
    return false;
}

void SerialCommunicationClass::processFrame(const uint8_t* data) {
    DataFrame frame{};

    memcpy(&frame.bullet_velocity, &data[4], sizeof(float));
    memcpy(&frame.gimbal_pitch, &data[8], sizeof(uint16_t));
    memcpy(&frame.gimbal_yaw_small, &data[10], sizeof(uint16_t));
    memcpy(&frame.mark, &data[12], sizeof(uint16_t));
    memcpy(&frame.color, &data[14], sizeof(uint8_t));
    memcpy(&frame.z_rotation_velocity, &data[15], sizeof(int16_t));
    // 哨兵新增
    memcpy(&frame.chassis_mode, &data[17], sizeof(uint8_t));
    memcpy(&frame.lack_blood_son_mode, &data[18], sizeof(uint8_t));
    memcpy(&frame.gimbal_yaw_big, &data[19], sizeof(int16_t));
    memcpy(frame.all_car_rebirth_infos, &data[21], sizeof(uint8_t) * 4);

    // 格式化输出
    RCLCPP_INFO(node->get_logger(), 
        "\033[1;34m[Received Data]\033[0m\n"
        "\033[1;32mBullet Velocity:\033[0m %.2f m/s\n"
        "\033[1;32mBullet Angle:\033[0m %d\n"
        "\033[1;33mGimbal Yaw:\033[0m %d\n"
        "\033[1;36mMark:\033[0m %d\n"
        "\033[1;31mColor:\033[0m %d\n"
        "\033[1;35mZ Rotation Velocity:\033[0m %d\n"
        "\033[1;34mchassis_mode:\033[0m %d\n"
        "\033[1;34mlack_blood_son_mode:\033[0m %d\n"
        "\033[1;34mgimbal_yaw_big:\033[0m %d\n",
        frame.bullet_velocity,
        frame.gimbal_pitch,
        frame.gimbal_yaw_small,
        frame.mark,
        frame.color,
        frame.z_rotation_velocity,
        frame.chassis_mode,
        frame.lack_blood_son_mode,
        frame.gimbal_yaw_big
    );

    SerialData msg;
    msg.bullet_velocity = frame.bullet_velocity;
    msg.gimbal_pitch = static_cast<float>(frame.gimbal_pitch - 32844) * (60.0f / 173.5f * M_PI / 180.0f) + 0.40;
    msg.gimbal_yaw_small = static_cast<float>(frame.gimbal_yaw_small - 1350) * M_PI / 4096.0f;
    msg.gimbal_yaw_big = - static_cast<float>(frame.gimbal_yaw_big) * M_PI / 4096.0f;
    msg.color = frame.color;

    msg.gimbal_pitch = std::atan2(std::sin(msg.gimbal_pitch), std::cos(msg.gimbal_pitch));
    msg.gimbal_yaw_small = std::atan2(std::sin(msg.gimbal_yaw_small), std::cos(msg.gimbal_yaw_small));
    msg.gimbal_yaw_big = std::atan2(std::sin(msg.gimbal_yaw_big), std::cos(msg.gimbal_yaw_big));

    msg.all_car_rebirth_infos.resize(4);
    for (int i=0; i<4; i++) {
        msg.all_car_rebirth_infos[i] = frame.all_car_rebirth_infos[i];
    }
    
    serialDataCallback(msg);

    last_received_time = std::chrono::steady_clock::now();
}

void SerialCommunicationClass::processBuffer() {
    
    // 每次处理最多处理10个帧，防止处理过多数据导致阻塞
    static const size_t MAX_FRAMES_PER_LOOP = 10;
    size_t frames_processed = 0;

    while (buffer_index_ >= FRAME_MIN_SIZE && frames_processed < MAX_FRAMES_PER_LOOP) {
        // 安全检查：如果缓冲区接近满，立即清空
        if (buffer_index_ >= BUFFER_SIZE - 128) {
            RCLCPP_WARN(node->get_logger(), "Buffer approaching capacity (%zu bytes), clearing", buffer_index_);
            buffer_index_ = 0;
            return;
        }

        // 查找帧头
        size_t header_pos = 0;
        bool found_header = false;
        
        // 只在合理范围内查找帧头
        while (header_pos <= buffer_index_ - 3 && header_pos < 128) {
            if (buffer_[header_pos] == FRAME_HEADER1 && 
                buffer_[header_pos + 1] == FRAME_HEADER2 && 
                buffer_[header_pos + 2] == COMMAND_CODE) {
                found_header = true;
                break;
            }
            ++header_pos;
        }

        if (!found_header) {
            // 如果找不到帧头，清空缓冲区
            buffer_index_ = 0;
            return;
        }

        // 如果帧头前有无效数据，移除它们
        if (header_pos > 0) {
            if (header_pos < buffer_index_) {
                memmove(buffer_.data(), buffer_.data() + header_pos, buffer_index_ - header_pos);
                buffer_index_ -= header_pos;
            } else {
                buffer_index_ = 0;
                return;
            }
        }

        // 检查是否有完整的帧
        if (buffer_index_ < 4) {
            return;  // 等待更多数据
        }

        uint8_t data_length = buffer_[3];
        size_t frame_length = data_length + 5;

        // 验证帧长度的合理性
        if (data_length > 64 || frame_length > BUFFER_SIZE) {  // 假设最大帧长度为64字节
            RCLCPP_ERROR(node->get_logger(), "Invalid frame length detected: %zu", frame_length);
            buffer_index_ = 0;
            return;
        }

        if (buffer_index_ < frame_length) {
            return;  // 等待完整帧
        }

        RCLCPP_DEBUG(node->get_logger(), "frame_length: %ld", frame_length);
        RCLCPP_DEBUG(node->get_logger(), "received crc: %d", buffer_[frame_length - 1]);
        RCLCPP_DEBUG(node->get_logger(), "target crc: %d", CRC8_Check_Sum(buffer_.data(), frame_length - 1));
        // CRC校验
        if (CRC8_Check_Sum(buffer_.data(), frame_length - 1) == buffer_[frame_length - 1]) {
            processFrame(buffer_.data());
            frames_processed++;
        } else {
            // CRC错误，移除这一帧
            RCLCPP_WARN(node->get_logger(), "CRC check failed, discarding frame");
            memmove(buffer_.data(), buffer_.data() + 3, buffer_index_ - 3);
            buffer_index_ -= 3;
            continue;
        }

        // 移除已处理的帧
        if (frame_length < buffer_index_) {
            memmove(buffer_.data(), buffer_.data() + frame_length, buffer_index_ - frame_length);
            buffer_index_ -= frame_length;
        } else {
            buffer_index_ = 0;
        }
    }

    // 如果还有数据未处理，在下一个循环继续处理
    if (buffer_index_ >= FRAME_MIN_SIZE) {
        RCLCPP_DEBUG(node->get_logger(), "Remaining data in buffer: %zu bytes", buffer_index_);
    }
}

void SerialCommunicationClass::timerCallback() {
    // 检查串口状态
    if (fd_ < 0) {
        if (std::chrono::steady_clock::now() - last_reconnect_time > std::chrono::seconds(3)) {
            RCLCPP_ERROR(node->get_logger(), "Serial port not available, trying reconnect");
            tryReconnect();
        }
        return;
    }
    if (std::chrono::steady_clock::now() - last_received_time > std::chrono::seconds(3)) {
        if (std::chrono::steady_clock::now() - last_reconnect_time > std::chrono::seconds(3)) {
            RCLCPP_ERROR(node->get_logger(), "No data received, trying reconnect");
            tryReconnect();
        }
        return;
    }

    // 读取串口数据
    if (buffer_index_ < BUFFER_SIZE - 128) {
        uint8_t temp_buffer[128];
        ssize_t bytes_read = read(fd_, temp_buffer, sizeof(temp_buffer));
        
        if (bytes_read > 0) {
            if (buffer_index_ + bytes_read < BUFFER_SIZE) {
                memcpy(buffer_.data() + buffer_index_, temp_buffer, bytes_read);
                buffer_index_ += bytes_read;
                processBuffer();
            } else {
                RCLCPP_WARN(node->get_logger(), "Buffer near full, discarding data");
                buffer_index_ = 0;
            }
        }
    }
}

void SerialCommunicationClass::timerThread() {
    while (running) {
        auto start = std::chrono::steady_clock::now();

        timerCallback();

        // 休眠至下一次调用
        std::this_thread::sleep_until(start + std::chrono::microseconds(1000));  // 大约1ms周期
    }
}
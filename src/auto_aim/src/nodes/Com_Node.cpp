// Com_Node.cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
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
#include "armor_detector/CRC.h"
#include "auto_aim/msg/serial_data.hpp"
#include "auto_aim/msg/gimbal_command.hpp"
#include <dirent.h>  // 用于遍历/dev目录
#include <sys/types.h>
#include <sys/stat.h>
class SerialCommunicationNode : public rclcpp::Node {
public:
    SerialCommunicationNode() : Node("serial_communication_node"), fd_(-1),last_debug_time_(this->now())  {
        // 使用可靠性QoS配置
        rclcpp::QoS qos(100);
        qos.reliable();
        qos.durability_volatile();
        
        serial_data_pub_ = this->create_publisher<auto_aim::msg::SerialData>(
            "serial_data", qos);
            
        gimbal_command_sub_ = this->create_subscription<auto_aim::msg::GimbalCommand>(
            "gimbal_command", qos,
            std::bind(&SerialCommunicationNode::gimbalCommandCallback, this, std::placeholders::_1));

        initializeSerial();
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&SerialCommunicationNode::timerCallback, this)
        );
    }

        ~SerialCommunicationNode() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

private:
    rclcpp::Publisher<auto_aim::msg::SerialData>::SharedPtr serial_data_pub_;
    rclcpp::Subscription<auto_aim::msg::GimbalCommand>::SharedPtr gimbal_command_sub_;
    static constexpr size_t BUFFER_SIZE = 1024;
    static constexpr uint8_t FRAME_HEADER1 = 0x42;
    static constexpr uint8_t FRAME_HEADER2 = 0x52;
    static constexpr uint8_t COMMAND_CODE = 0xCD;
    static constexpr size_t FRAME_MIN_SIZE = 5;
    std::queue<auto_aim::msg::GimbalCommand::SharedPtr> command_queue_;
    std::mutex queue_mutex_;
    static constexpr size_t MAX_QUEUE_SIZE = 10;
    struct DataFrame {
        float bullet_velocity;
        float bullet_angle;
        int16_t gimbal_yaw;
        uint16_t mark;
        uint8_t color;
        float z_rotation_velocity;
    };

    int fd_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::array<uint8_t, BUFFER_SIZE> buffer_;
    size_t buffer_index_ = 0;
    std::atomic<int> received_commands_count_{0};
    std::atomic<int> sent_commands_count_{0};
    rclcpp::Time last_debug_time_;
    
    
    void initializeSerial() {
        std::string port = findAvailableSerialPort();
        if (port.empty()) {
            RCLCPP_ERROR(get_logger(), "No available serial port found!");
            return;
        }

        int baudrate = this->declare_parameter<int>("baudrate", 256000);
        fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ < 0) {
            RCLCPP_ERROR(get_logger(), "Failed to open port %s: %s", port.c_str(), strerror(errno));
            return;
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));



        if (tcgetattr(fd_, &tty) != 0) {
            RCLCPP_ERROR(get_logger(), "Failed to get serial attributes");
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
            RCLCPP_ERROR(get_logger(), "Failed to set serial attributes");
            close(fd_);
            fd_ = -1;
            return;
        }

        tcflush(fd_, TCIOFLUSH);
        RCLCPP_INFO(get_logger(), "Serial initialized: %s @ %d", port.c_str(), baudrate);
    }

        // 查找可用的串口
    std::string findAvailableSerialPort() {
        struct dirent *entry;
        DIR *dp = opendir("/dev/");
        if (dp == nullptr) {
            RCLCPP_ERROR(get_logger(), "Failed to open /dev/ directory");
            return "";
        }

        std::string port;
        while ((entry = readdir(dp)) != nullptr) {
            if (strncmp(entry->d_name, "ttyACM", 6) == 0) {  // 匹配ttyACM串口
                std::string candidate_port = "/dev/" + std::string(entry->d_name);
                int fd = open(candidate_port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
                if (fd >= 0) {
                    close(fd);  // 串口可用，返回串口名称
                    port = candidate_port;
                    break;
                }
            }
        }

        closedir(dp);
        return port;
    }

    bool sendData(float pitch, float yaw) {
        // 总大小 = 帧头(2) + 命令码(1) + 长度(1) + pitch(4) + yaw(2) + CRC(1) = 11字节
        std::array<uint8_t, 11> tx_data{};
        
        tx_data[0] = FRAME_HEADER1;
        tx_data[1] = FRAME_HEADER2;
        tx_data[2] = COMMAND_CODE;
        tx_data[3] = 0x06;  // 数据长度为6（4字节pitch + 2字节yaw）
        
        // 处理pitch (4字节)
        memcpy(&tx_data[4], &pitch, sizeof(float));  // 4字节float
        
        // 处理yaw (2字节)
        int16_t yaw_int16 = static_cast<int16_t>(yaw * 100);  // 将float转换为定点数
        memcpy(&tx_data[8], &yaw_int16, sizeof(int16_t));  // 2字节
        
        // 计算并添加CRC
        tx_data[10] = CRC8_Check_Sum(tx_data.data(), 10);

        ssize_t written = write(fd_, tx_data.data(), tx_data.size());
        if (written == static_cast<ssize_t>(tx_data.size())) {
            RCLCPP_DEBUG(get_logger(), "TX: pitch=%.2f yaw=%.2f(int16=%d)", 
                        pitch, yaw, yaw_int16);
            return true;
        }
        return false;
    }

    void processFrame(const uint8_t* data, size_t length) {
        if (length < FRAME_MIN_SIZE || data[2] != COMMAND_CODE || data[3] != 0x0F) {
            return;
        }

        DataFrame frame{};
        size_t offset = 4;

        memcpy(&frame.bullet_velocity, &data[offset], sizeof(float));
        offset += sizeof(float);
        memcpy(&frame.bullet_angle, &data[offset], sizeof(float));
        offset += sizeof(float);
        memcpy(&frame.gimbal_yaw, &data[offset], sizeof(int16_t));
        offset += sizeof(int16_t);
        memcpy(&frame.mark, &data[offset], sizeof(uint16_t));
        offset += sizeof(uint16_t);
        memcpy(&frame.color, &data[offset], sizeof(uint8_t));
        offset += sizeof(uint8_t);
        memcpy(&frame.z_rotation_velocity, &data[offset], sizeof(float));

        // 格式化输出
        RCLCPP_INFO(get_logger(), 
            "\033[1;34m[Received Data]\033[0m\n"
            "\033[1;32mBullet Velocity:\033[0m %.2f m/s\n"
            "\033[1;32mBullet Angle:\033[0m %.2f°\n"
            "\033[1;33mGimbal Yaw:\033[0m %.2f°\n"
            "\033[1;36mMark:\033[0m %d\n"
            "\033[1;31mColor:\033[0m %d\n"
            "\033[1;35mZ Rotation Velocity:\033[0m %.2f rad/s",
            frame.bullet_velocity,
            frame.bullet_angle,
            frame.gimbal_yaw  / 100.0f,
            frame.mark,
            frame.color,
            frame.z_rotation_velocity

        
        );
        auto msg = auto_aim::msg::SerialData();
        msg.bullet_velocity = frame.bullet_velocity;
        msg.bullet_angle = frame.bullet_angle;
        msg.gimbal_yaw = frame.gimbal_yaw / 100.0f;
        msg.color = frame.color;
        
        // 使用异步发布
        serial_data_pub_->publish(std::move(msg));
    }
    void gimbalCommandCallback(const auto_aim::msg::GimbalCommand::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        received_commands_count_++;
        
        RCLCPP_INFO(get_logger(), 
            "[Command Received] Pitch: %.2f, Yaw: %.2f, Queue Size: %zu", 
            msg->pitch, msg->yaw, command_queue_.size());

        if (command_queue_.size() < MAX_QUEUE_SIZE) {
            command_queue_.push(msg);
        } else {
            // 队列满时，删除最旧的命令
            command_queue_.pop();
            command_queue_.push(msg);
            RCLCPP_WARN(get_logger(), "Command queue full, dropping oldest command");
        }
    }

    void processBuffer() {
        // 每次处理最多处理5个帧，防止处理过多数据导致阻塞
        static const size_t MAX_FRAMES_PER_LOOP = 5;
        size_t frames_processed = 0;

        while (buffer_index_ >= FRAME_MIN_SIZE && frames_processed < MAX_FRAMES_PER_LOOP) {
            // 安全检查：如果缓冲区接近满，立即清空
            if (buffer_index_ >= BUFFER_SIZE - 128) {
                RCLCPP_WARN(get_logger(), "Buffer approaching capacity (%zu bytes), clearing", buffer_index_);
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
                RCLCPP_ERROR(get_logger(), "Invalid frame length detected: %zu", frame_length);
                buffer_index_ = 0;
                return;
            }

            if (buffer_index_ < frame_length) {
                return;  // 等待完整帧
            }

            // CRC校验
            if (CRC8_Check_Sum(buffer_.data(), frame_length - 1) == buffer_[frame_length - 1]) {
                processFrame(buffer_.data(), frame_length);
                frames_processed++;
            } else {
                // CRC错误，移除这一帧
                RCLCPP_WARN(get_logger(), "CRC check failed, discarding frame");
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
            RCLCPP_DEBUG(get_logger(), "Remaining data in buffer: %zu bytes", buffer_index_);
        }
    }

    void timerCallback() {
        // 检查串口状态
        if (fd_ < 0) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Serial port not available");
            return;
        }

        auto current_time = this->now();
        
        // 统计信息部分保持不变
        if ((current_time - last_debug_time_).seconds() >= 1.0) {
            RCLCPP_INFO(get_logger(),
                "Statistics: Received %d commands, Sent %d commands, Queue size: %zu",
                received_commands_count_.load(),
                sent_commands_count_.load(),
                command_queue_.size());
            
            received_commands_count_ = 0;
            sent_commands_count_ = 0;
            last_debug_time_ = current_time;
        }

        // 读取串口数据部分保持不变
        if (buffer_index_ < BUFFER_SIZE - 128) {
            uint8_t temp_buffer[128];
            ssize_t bytes_read = read(fd_, temp_buffer, sizeof(temp_buffer));
            
            if (bytes_read > 0) {
                if (buffer_index_ + bytes_read < BUFFER_SIZE) {
                    memcpy(buffer_.data() + buffer_index_, temp_buffer, bytes_read);
                    buffer_index_ += bytes_read;
                    processBuffer();
                } else {
                    RCLCPP_WARN(get_logger(), "Buffer near full, discarding data");
                    buffer_index_ = 0;
                }
            }
        }

        // 修改发送命令部分
        static rclcpp::Time last_send_time = this->now();
        double elapsed = (current_time - last_send_time).seconds();
        
        // Debug: 打印串口状态和时间间隔
        RCLCPP_INFO(get_logger(), "Serial fd: %d, Elapsed time: %.6f seconds", fd_, elapsed);
        
        // 提高发送频率到200Hz（5ms间隔）
        if (elapsed >= 0.005) {
            // Debug: 打印进入发送逻辑
            RCLCPP_INFO(get_logger(), "Entering send logic");
            
            {  // 使用花括号限制锁的范围
                std::lock_guard<std::mutex> lock(queue_mutex_);
                // Debug: 打印队列状态
                RCLCPP_INFO(get_logger(), "Queue size before sending: %zu", command_queue_.size());
                
                while (!command_queue_.empty()) {  // 处理队列中的所有命令
                    auto cmd = command_queue_.front();
                    // Debug: 打印即将发送的命令
                    RCLCPP_INFO(get_logger(), "Attempting to send - Pitch: %.2f, Yaw: %.2f", 
                            cmd->pitch, cmd->yaw);
                    
                    bool send_success = sendData(cmd->pitch, cmd->yaw);
                    
                    // Debug: 打印发送结果
                    RCLCPP_INFO(get_logger(), "Send result: %s", send_success ? "Success" : "Failed");
                    
                    if (send_success) {
                        command_queue_.pop();
                        sent_commands_count_++;
                        RCLCPP_INFO(get_logger(),
                            "[Command Sent] Pitch: %.2f, Yaw: %.2f, Queue Size: %zu",
                            cmd->pitch, cmd->yaw, command_queue_.size());
                    } else {
                        RCLCPP_WARN(get_logger(), "Failed to send command");
                        break;  // 如果发送失败，暂停发送
                    }
                    
                    // 检查是否超过最大处理时间
                    double process_time = (this->now() - current_time).seconds();
                    // Debug: 打印处理时间
                    RCLCPP_INFO(get_logger(), "Process time: %.6f seconds", process_time);
                    
                    if (process_time > 0.001) {  // 最多处理1ms
                        RCLCPP_INFO(get_logger(), "Breaking due to time limit");
                        break;
                    }
                }
                // Debug: 打印队列最终状态
                RCLCPP_INFO(get_logger(), "Queue size after sending: %zu", command_queue_.size());
            }
            last_send_time = current_time;
        } else {
            // Debug: 打印未达到发送间隔的情况
            RCLCPP_INFO(get_logger(), "Skipping send due to time interval (%.6f < 0.005)", elapsed);
        }
    }
};
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SerialCommunicationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

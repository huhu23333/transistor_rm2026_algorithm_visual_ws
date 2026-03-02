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
#include <string>
#include <libudev.h>
#include <thread>

std::string getSerialPortInfo(const std::string& port) {
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
        const char *manufacturer = udev_device_get_sysattr_value(parent, "manufacturer");
        const char *serial = udev_device_get_sysattr_value(parent, "serial");
        
        if (product) {
            result += "Product: " + std::string(product) + "\n";
        }
        if (manufacturer) {
            result += "Manufacturer: " + std::string(manufacturer) + "\n";
        }
        if (serial) {
            result += "Serial: " + std::string(serial) + "\n";
        }
        
        // 获取vendor和product ID
        const char *idVendor = udev_device_get_sysattr_value(parent, "idVendor");
        const char *idProduct = udev_device_get_sysattr_value(parent, "idProduct");
        if (idVendor && idProduct) {
            result += "VID:PID = " + std::string(idVendor) + ":" + std::string(idProduct) + "\n";
        }
    }
    
    udev_device_unref(dev);
    udev_unref(udev);
    return result;
}


struct SerialData {
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

class SerialCommunicationClass {
public:
    SerialCommunicationClass(std::function<void(const SerialData&)> serialDataCallback);
    ~SerialCommunicationClass();
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

    std::function<void(const SerialData&)> serialDataCallback;
    bool running = true;

    std::chrono::steady_clock::time_point last_reconnect_time;
    std::chrono::steady_clock::time_point last_received_time;
    
    void initializeSerial();
    std::vector<std::string> findAvailableSerialPorts();
    void processFrame(const uint8_t* data);
    void processBuffer();
    void tryReconnect();
};

uint32_t HAL_CRC_Calculate(const uint8_t* data, size_t length);

SerialCommunicationClass::SerialCommunicationClass(std::function<void(const SerialData&)> serialDataCallback) 
: serialDataCallback(serialDataCallback), fd_(-1) {
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
    std::string port = ports[0];

    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd_ < 0) {
        printf("Failed to open port %s: %s\n", port.c_str(), strerror(errno));
        return;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));

    if (tcgetattr(fd_, &tty) != 0) {
        printf("Failed to get serial attributes\n");
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
        printf("Failed to set serial attributes\n");
        close(fd_);
        fd_ = -1;
        return;
    }

    tcflush(fd_, TCIOFLUSH);
    printf("Serial initialized: %s\n", port.c_str());
    printf("%s\n", getSerialPortInfo(port.substr(5)).c_str());
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
                break;
            }
        }
    }

    closedir(dp);
    return ports;
}

void SerialCommunicationClass::processFrame(const uint8_t* data) {
    DataFrame frame{};

    memcpy(&frame.header1, &data[0], sizeof(uint8_t));
    memcpy(&frame.header2, &data[1], sizeof(uint8_t));
    memcpy(&frame.header3, &data[2], sizeof(uint8_t));
    memcpy(&frame.data_len, &data[3], sizeof(uint8_t));
    memcpy(&frame.rectified_avx_multiply_8, &data[4], sizeof(int32_t));
    memcpy(&frame.rectified_avy_multiply_8, &data[8], sizeof(int32_t));
    memcpy(&frame.rectified_avz_multiply_8, &data[12], sizeof(int32_t));
    memcpy(&frame.received_ax, &data[16], sizeof(short));
    memcpy(&frame.received_ay, &data[18], sizeof(short));
    memcpy(&frame.received_az, &data[20], sizeof(short));
    memcpy(&frame.euler_yaw, &data[22], sizeof(float));
    memcpy(&frame.euler_pitch, &data[26], sizeof(float));
    memcpy(&frame.euler_roll, &data[30], sizeof(float));
    memcpy(&frame.crc32, &data[34], sizeof(uint32_t));

    SerialData msg;
    msg.rectified_avx_multiply_8 = frame.rectified_avx_multiply_8;
    msg.rectified_avy_multiply_8 = frame.rectified_avy_multiply_8;
    msg.rectified_avz_multiply_8 = frame.rectified_avz_multiply_8;
    msg.received_ax = frame.received_ax;
    msg.received_ay = frame.received_ay;
    msg.received_az = frame.received_az;
    msg.euler_yaw = frame.euler_yaw;
    msg.euler_pitch = frame.euler_pitch;
    msg.euler_roll = frame.euler_roll;
    
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
            printf("Buffer approaching capacity (%zu bytes), clearing\n", buffer_index_);
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
                buffer_[header_pos + 2] == FRAME_HEADER3) {
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
        size_t frame_length = data_length + 4 + 4;

        // 验证帧长度的合理性
        if (data_length > 64 || frame_length > BUFFER_SIZE) {  // 假设最大帧长度为64字节
            printf("Invalid frame length detected: %zu\n", frame_length);
            buffer_index_ = 0;
            return;
        }

        if (buffer_index_ < frame_length) {
            return;  // 等待完整帧
        }

        // CRC校验
        uint32_t computed_crc32 = HAL_CRC_Calculate(buffer_.data(), frame_length-4);
        uint32_t received_crc32;
        memcpy(&received_crc32, buffer_.data()+(frame_length-4), sizeof(uint32_t));
        if (computed_crc32 == received_crc32) {
            processFrame(buffer_.data());
            frames_processed++;
        } else {
            // CRC错误，移除这一帧
            printf("CRC check failed, discarding frame\n");
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
        printf("Remaining data in buffer: %zu bytes\n", buffer_index_);
    }
}

void SerialCommunicationClass::timerCallback() {
    // 检查串口状态
    if (fd_ < 0) {
        if (std::chrono::steady_clock::now() - last_reconnect_time > std::chrono::seconds(3)) {
            printf("Serial port not available, trying reconnect\n");
            tryReconnect();
        }
        return;
    }
    if (std::chrono::steady_clock::now() - last_received_time > std::chrono::seconds(3)) {
        if (std::chrono::steady_clock::now() - last_reconnect_time > std::chrono::seconds(3)) {
            printf("No data received, trying reconnect\n");
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
                printf("Buffer near full, discarding data\n");
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


















#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <memory>
#include <thread>

// ================= 常量定义 =================
const double TO_DEGREE_RATIO = 51016.0;
const double TO_RAD_RATIO = TO_DEGREE_RATIO * 180.0 / M_PI;
const double TO_MPERSQ_RATIO = 2048.0 / 9.81;

// ================= 工具函数 =================
uint32_t HAL_CRC_Calculate(const uint8_t* data, size_t length) {
    const uint32_t POLYNOMIAL = 0x04C11DB7;
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; i += 4) {
        uint32_t word = 0;
        if (i + 3 < length) {
            word = *reinterpret_cast<const uint32_t*>(&data[i]);
        } else {
            // 处理不足4字节的情况
            for (size_t j = 0; j < 4 && i + j < length; ++j) {
                word |= static_cast<uint32_t>(data[i + j]) << (8 * j);
            }
        }
        
        crc ^= word;
        
        for (int j = 0; j < 32; ++j) {
            if (crc & 0x80000000) {
                crc = (crc << 1) ^ POLYNOMIAL;
            } else {
                crc <<= 1;
            }
        }
    }
    
    return crc;
}

// ================= RestFrame 类 =================
class RestFrame {
private:
    double camera_yaw, camera_pitch, camera_roll;
    double camera_x, camera_y, camera_z;
    
public:
    RestFrame() : camera_yaw(0), camera_pitch(0), camera_roll(0), 
                  camera_x(0), camera_y(0), camera_z(0) {}
    
    void updateCamOrientation(float yaw, float pitch, float roll) {
        camera_yaw = yaw;
        camera_pitch = pitch;
        camera_roll = roll;
    }
    
    void updateCamPosition(float x, float y, float z) {
        camera_x = x;
        camera_y = y;
        camera_z = z;
    }
    
    cv::Point3f worldToPnpP3f(const cv::Point3f& world_pos) {
        // 使用旋转矩阵方法
        Eigen::Matrix3d R = eulerToRotationMatrix(camera_yaw, camera_pitch, camera_roll);
        Eigen::Vector3d v(world_pos.x - camera_x, 
                         world_pos.y - camera_y, 
                         world_pos.z - camera_z);
        Eigen::Vector3d v_rotated = R.transpose() * v;
        
        // 转换为PNP坐标系 (x, -z, y)
        return cv::Point3f(v_rotated.x(), -v_rotated.z(), v_rotated.y());
    }
    
private:
    Eigen::Matrix3d eulerToRotationMatrix(double yaw, double pitch, double roll) {
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitY());
        
        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
        return q.toRotationMatrix();
    }
};


class Main {
public:
    Main() {
        serial_communication_ = std::make_shared<SerialCommunicationClass>(std::bind(&Main::serialDataCallback, this, std::placeholders::_1));
        com_timer_thread_ = std::thread(std::bind(&SerialCommunicationClass::timerThread, serial_communication_));
        // com_timer_thread_.detach();
        
        // 初始化相机内参和畸变系数
        camera_matrix = (cv::Mat_<double>(3, 3) << 
            1.31280460e+03, 0.00000000e+00, 6.38736364e+02,
            0.00000000e+00, 1.31309593e+03, 5.34133502e+02,
            0.00000000e+00, 0.00000000e+00, 1.00000000e+00);
        
        dist_coeffs = (cv::Mat_<double>(5, 1) << 
            -0.05392145, -0.02516686, -0.00222499, -0.00149047, 0.43693918);
        
        // 创建显示窗口
        cv::namedWindow("IMU 3D Visualization", cv::WINDOW_NORMAL);
        cv::resizeWindow("IMU 3D Visualization", 1280, 1024);
        
        last_time = std::chrono::steady_clock::now();
    }

    void start_loop() {
        while (true) {
        // 读取串口数据
            
            // 计算时间间隔（用于帧率显示）
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last_time).count();
            last_time = now;
            
            // 转换为角度
            double yaw_deg = euler_yaw * 180.0 / M_PI;
            double pitch_deg = euler_pitch * 180.0 / M_PI;
            double roll_deg = euler_roll * 180.0 / M_PI;
            
            // 更新相机姿态
            rest_frame.updateCamOrientation(euler_yaw, euler_pitch, euler_roll);
            
            // 打印姿态信息
            // std::cout << "\rYaw: " << yaw_deg << "°, Pitch: " << pitch_deg << "°, Roll: " << roll_deg 
            //           << "°, dt: " << dt * 1000 << "ms" << std::flush;
            
            // 创建空白图像
            cv::Mat frame = cv::Mat::zeros(1024, 1280, CV_8UC3);
            
            // 绘制网格点
            for (float x = -3.0f; x <= 3.0f; x += 0.2f) {
                for (float y = -3.0f; y <= 3.0f; y += 0.2f) {
                    float z = -0.5f; // 固定高度
                    
                    cv::Point3f world_point(x * 1000.0f, y * 1000.0f, z * 1000.0f);
                    cv::Point3f cam_point = rest_frame.worldToPnpP3f(world_point);
                    
                    // 投影到图像平面
                    std::vector<cv::Point3f> object_points = {cam_point};
                    std::vector<cv::Point2f> image_points;
                    
                    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
                    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
                    
                    cv::projectPoints(object_points, rvec, tvec, 
                                     camera_matrix, dist_coeffs, 
                                     image_points);
                    
                    cv::Point2f img_point = image_points[0];
                    
                    // 检查点是否在图像范围内
                    if (img_point.x >= 0 && img_point.x < 1280 && 
                        img_point.y >= 0 && img_point.y < 1024) {
                        cv::circle(frame, img_point, 2, cv::Scalar(200, 200, 200), -1);
                    }
                }
            }
            
            // 绘制坐标轴
            cv::Point3f origin(0, 0, 0);
            cv::Point3f x_axis(1000, 0, 0);
            cv::Point3f y_axis(0, 1000, 0);
            cv::Point3f z_axis(0, 0, 1000);
            
            // 将坐标轴点转换并投影
            auto projectPoint = [&](const cv::Point3f& world_pt) -> cv::Point2f {
                cv::Point3f cam_pt = rest_frame.worldToPnpP3f(world_pt);
                std::vector<cv::Point3f> pts = {cam_pt};
                std::vector<cv::Point2f> img_pts;
                cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
                cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
                cv::projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs, img_pts);
                return img_pts[0];
            };
            
            cv::Point2f origin_img = projectPoint(origin);
            cv::Point2f x_img = projectPoint(x_axis);
            cv::Point2f y_img = projectPoint(y_axis);
            cv::Point2f z_img = projectPoint(z_axis);
            
            // 绘制坐标轴
            if (origin_img.x >= 0 && origin_img.x < 1280 && origin_img.y >= 0 && origin_img.y < 1024) {
                if (x_img.x >= 0 && x_img.x < 1280 && x_img.y >= 0 && x_img.y < 1024) {
                    cv::arrowedLine(frame, origin_img, x_img, cv::Scalar(0, 0, 255), 3);
                    cv::putText(frame, "X", x_img, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                }
                if (y_img.x >= 0 && y_img.x < 1280 && y_img.y >= 0 && y_img.y < 1024) {
                    cv::arrowedLine(frame, origin_img, y_img, cv::Scalar(0, 255, 0), 3);
                    cv::putText(frame, "Y", y_img, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                }
                if (z_img.x >= 0 && z_img.x < 1280 && z_img.y >= 0 && z_img.y < 1024) {
                    cv::arrowedLine(frame, origin_img, z_img, cv::Scalar(255, 0, 0), 3);
                    cv::putText(frame, "Z", z_img, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
                }
            }
            
            // 显示姿态信息
            std::string info = "Yaw: " + std::to_string(static_cast<int>(yaw_deg)) + 
                             " Pitch: " + std::to_string(static_cast<int>(pitch_deg)) + 
                             " Roll: " + std::to_string(static_cast<int>(roll_deg));
            cv::putText(frame, info, cv::Point(20, 40), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            
            // 显示帧率
            double fps = 1.0 / dt;
            std::string fps_str = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(frame, fps_str, cv::Point(20, 80), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            
            // 显示图像
            cv::imshow("IMU 3D Visualization", frame);

            // 检查按键
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') { // ESC 或 q 键退出
                break;
                cv::destroyAllWindows();
            }
        }
    }
private:
    std::shared_ptr<SerialCommunicationClass> serial_communication_;
    std::thread com_timer_thread_;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    RestFrame rest_frame;
    std::chrono::steady_clock::time_point last_time;

    float euler_yaw;
    float euler_pitch;
    float euler_roll;

    void serialDataCallback(const SerialData& msg) {
        euler_yaw = msg.euler_yaw;
        euler_pitch = msg.euler_pitch;
        euler_roll = msg.euler_roll;
    }
};

// ================= 主函数 =================
int main() {
    Main().start_loop();
    
    return 0;
}

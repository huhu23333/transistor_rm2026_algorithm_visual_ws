#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <unistd.h>
#include "MvCameraControl.h"

extern bool g_bExit;
extern cv::Mat g_image;
extern pthread_mutex_t g_mutex;

class Camera {
public:
    void* handle;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    unsigned int nPayloadSize;

    // 构造函数：初始化摄像头
    Camera(const std::string& deviceIp, const std::string& netIp);
    
    // 析构函数：释放资源
    ~Camera();

    // IP地址解析函数
    static void parseIp(const std::string& ip, unsigned int& parsedIp);

    // 摄像头取流线程
    static void* workThread(void* pUser);

    // 新增：设置曝光时间（单位：微秒）
    bool setExposureTime(float exposureTime);

    // 新增：设置增益值（范围通常在0-15之间）
    bool setGain(float gain);

private:
    // 新增：初始化相机参数
    bool initCameraParams();
};

#endif // CAMERA_H

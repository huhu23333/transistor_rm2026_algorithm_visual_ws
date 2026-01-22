#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <unistd.h>
#include "MvCameraControl.h"

extern bool g_bExit;
extern cv::Mat g_image;
extern pthread_mutex_t g_mutex;

enum CameraType {
    GIGE_CAMERA,
    USB_CAMERA
};

class Camera {
public:
    void* handle;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    unsigned int nPayloadSize;
    // GigE相机构造函数
    Camera(const std::string& deviceIp, const std::string& netIp);
    
    // USB相机构造函数
    Camera(int deviceIndex = 0);
    
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
    
    // 枚举USB设备
    static std::vector<std::string> enumUSBDevices();

private:
    CameraType cameraType;
    
    // 新增：初始化相机参数
    bool initCameraParams();
    bool initCameraCommonParams();
    bool startGrabbing();
    
    // 图像处理
    bool processImage(unsigned char* pData, MV_FRAME_OUT_INFO_EX& stImageInfo, cv::Mat& outputImage);
};

#endif // CAMERA_H

#include "camera/Camera.h"

bool g_bExit = false;
cv::Mat g_image;
pthread_mutex_t g_mutex;

bool Camera::initCameraParams() {
    int nRet;
    
    // 设置默认曝光时间 (5000微秒)
    nRet = MV_CC_SetFloatValue(handle, "ExposureTime", 5100.0f);
    if (MV_OK != nRet) {
        std::cerr << "Set ExposureTime fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Default exposure time set to 5000us" << std::endl;

    // 设置默认增益值 (10.0)
    nRet = MV_CC_SetFloatValue(handle, "Gain", 16.0f);
    if (MV_OK != nRet) {
        std::cerr << "Set Gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Default gain set to 10.0" << std::endl;

    return true;
}

bool Camera::setExposureTime(float exposureTime) {
    int nRet = MV_CC_SetFloatValue(handle, "ExposureTime", exposureTime);
    if (MV_OK != nRet) {
        std::cerr << "Set ExposureTime fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Exposure time set to " << exposureTime << "us" << std::endl;
    return true;
}

bool Camera::setGain(float gain) {
    int nRet = MV_CC_SetFloatValue(handle, "Gain", gain);
    if (MV_OK != nRet) {
        std::cerr << "Set Gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Gain set to " << gain << std::endl;
    return true;
}

// GigE相机构造函数
Camera::Camera(const std::string& deviceIp, const std::string& netIp) {
    int nRet = MV_OK;
    cameraType = GIGE_CAMERA;

    // 初始化SDK
    nRet = MV_CC_Initialize();
    if (MV_OK != nRet) {
        std::cerr << "Initialize SDK fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "SDK initialized successfully." << std::endl;
    }

    MV_CC_DEVICE_INFO stDevInfo;
    MV_GIGE_DEVICE_INFO stGigEDev;
    memset(&stDevInfo, 0, sizeof(MV_CC_DEVICE_INFO));
    memset(&stGigEDev, 0, sizeof(MV_GIGE_DEVICE_INFO));

    // 解析IP地址
    parseIp(deviceIp, stGigEDev.nCurrentIp);
    parseIp(netIp, stGigEDev.nNetExport);
    std::cout << "Device IP and Net IP parsed successfully." << std::endl;

    stDevInfo.nTLayerType = MV_GIGE_DEVICE;
    stDevInfo.SpecialInfo.stGigEInfo = stGigEDev;

    // 创建句柄
    handle = NULL;
    nRet = MV_CC_CreateHandle(&handle, &stDevInfo);
    if (MV_OK != nRet) {
        std::cerr << "Create Handle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "Handle created successfully." << std::endl;
    }

    // 打开设备
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet) {
        std::cerr << "Open device fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "Device opened successfully." << std::endl;
    }

    // 获取 GigE 相机的最佳数据包大小
    int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
    if (nPacketSize > 0) {
        nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
        if (MV_OK != nRet) {
            std::cerr << "Set Packet Size fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        } else {
            std::cout << "Optimal packet size set successfully." << std::endl;
        }
    }

    // 初始化相机参数
    if (!initCameraCommonParams()) {
        std::cerr << "Failed to initialize camera parameters!" << std::endl;
        exit(1);
    }

    // 开始取流
    if (!startGrabbing()) {
        std::cerr << "Failed to start grabbing!" << std::endl;
        exit(1);
    }
}

// USB相机构造函数
Camera::Camera(int deviceIndex) {
    int nRet = MV_OK;
    cameraType = USB_CAMERA;

    // 初始化SDK
    nRet = MV_CC_Initialize();
    if (MV_OK != nRet) {
        std::cerr << "Initialize SDK fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "SDK initialized successfully." << std::endl;
    }

    // 枚举USB设备
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet) {
        std::cerr << "Enum USB devices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    }

    if (stDeviceList.nDeviceNum == 0) {
        std::cerr << "No USB camera found!" << std::endl;
        exit(1);
    }

    if (deviceIndex >= stDeviceList.nDeviceNum) {
        std::cerr << "Device index out of range! Found " << stDeviceList.nDeviceNum << " devices." << std::endl;
        exit(1);
    }

    std::cout << "Found " << stDeviceList.nDeviceNum << " USB cameras, using device index: " << deviceIndex << std::endl;

    // 创建句柄
    handle = NULL;
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[deviceIndex]);
    if (MV_OK != nRet) {
        std::cerr << "Create Handle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "Handle created successfully." << std::endl;
    }

    // 打开设备
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet) {
        std::cerr << "Open device fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "Device opened successfully." << std::endl;
    }

    // 初始化相机参数
    if (!initCameraCommonParams()) {
        std::cerr << "Failed to initialize camera parameters!" << std::endl;
        exit(1);
    }

    // 开始取流
    if (!startGrabbing()) {
        std::cerr << "Failed to start grabbing!" << std::endl;
        exit(1);
    }
}

// 通用相机参数初始化
bool Camera::initCameraCommonParams() {
    int nRet;
    
    // 禁用自动曝光
    nRet = MV_CC_SetEnumValue(handle, "ExposureAuto", 0);
    if (MV_OK != nRet) {
        std::cerr << "Disable auto exposure fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Auto exposure disabled." << std::endl;

    // 禁用自动增益
    nRet = MV_CC_SetEnumValue(handle, "GainAuto", 0);
    if (MV_OK != nRet) {
        std::cerr << "Disable auto gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Auto gain disabled." << std::endl;

    // 禁用自动白平衡
    nRet = MV_CC_SetEnumValue(handle, "BalanceWhiteAuto", 0);
    if (MV_OK != nRet) {
        std::cerr << "Disable auto white balance fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Auto white balance disabled." << std::endl;

    // 设置相机参数
    if (!initCameraParams()) {
        std::cerr << "Failed to initialize camera exposure and gain parameters!" << std::endl;
        return false;
    }

    // 获取数据包大小
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    if (MV_OK != nRet) {
        std::cerr << "Get PayloadSize fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    } else {
        std::cout << "Payload size: " << stParam.nCurValue << std::endl;
    }
    nPayloadSize = stParam.nCurValue;

    return true;
}

// 开始取流
bool Camera::startGrabbing() {
    int nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet) {
        std::cerr << "Start grabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    } else {
        std::cout << "Grabbing started successfully." << std::endl;
    }

    // 启动取流线程
    pthread_t grabThread;
    pthread_create(&grabThread, NULL, workThread, this);
    std::cout << "Grabbing thread started." << std::endl;

    return true;
}

Camera::~Camera() {
    // 停止取流
    MV_CC_StopGrabbing(handle);
    std::cout << "Grabbing stopped." << std::endl;

    // 关闭设备
    MV_CC_CloseDevice(handle);
    std::cout << "Device closed." << std::endl;

    // 销毁句柄
    MV_CC_DestroyHandle(handle);
    std::cout << "Handle destroyed." << std::endl;

    // 释放SDK资源
    MV_CC_Finalize();
    std::cout << "SDK finalized." << std::endl;
}

void Camera::parseIp(const std::string& ip, unsigned int& parsedIp) {
    int parts[4];
    sscanf(ip.c_str(), "%d.%d.%d.%d", &parts[0], &parts[1], &parts[2], &parts[3]);
    parsedIp = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3];
    std::cout << "Parsed IP: " << ip << std::endl;
}

void* Camera::workThread(void* pUser) {
    Camera* pCam = (Camera*)pUser;
    MV_FRAME_OUT_INFO_EX stImageInfo;
    memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));

    unsigned char* pData = (unsigned char*)malloc(pCam->nPayloadSize);
    if (pData == NULL) {
        std::cerr << "Allocate memory fail!" << std::endl;
        return NULL;
    } else {
        std::cout << "Memory allocated for frame data." << std::endl;
    }

    while (!g_bExit) {
        int nRet = MV_CC_GetOneFrameTimeout(pCam->handle, pData, pCam->nPayloadSize, &stImageInfo, 5000);
        if (nRet == MV_OK) {
            // 检查帧数据完整性
            if (stImageInfo.nFrameLen != pCam->nPayloadSize) {
                std::cerr << "Frame data length mismatch! Expected: " << pCam->nPayloadSize << ", Received: " << stImageInfo.nFrameLen << std::endl;
                continue;
            }

            // 处理不同像素格式
            cv::Mat processedImage;
            if (!pCam->processImage(pData, stImageInfo, processedImage)) {
                std::cerr << "Image processing failed!" << std::endl;
                continue;
            }

            // 线程锁定，更新图像
            pthread_mutex_lock(&g_mutex);
            g_image = processedImage.clone();
            pthread_mutex_unlock(&g_mutex);
        } else {
            std::cerr << "Get frame timeout or error! nRet [0x" << std::hex << nRet << "]" << std::endl;
        }

        if (g_bExit) {
            break;
        }
    }

    free(pData);
    std::cout << "Frame grabbing thread exiting." << std::endl;
    return NULL;
}

bool Camera::processImage(unsigned char* pData, MV_FRAME_OUT_INFO_EX& stImageInfo, cv::Mat& outputImage) {
    switch (stImageInfo.enPixelType) {
        case PixelType_Gvsp_BayerGB8:  // GigE相机常见的Bayer格式
        case PixelType_Gvsp_BayerRG8:  // USB相机可能的Bayer格式
        case PixelType_Gvsp_BayerGR8:
        case PixelType_Gvsp_BayerBG8: {
            cv::Mat img(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
            cv::Mat bgrImg;
            
            // 根据不同的Bayer格式进行转换
            int conversionCode = -1;
            switch (stImageInfo.enPixelType) {
                case PixelType_Gvsp_BayerGB8:
                    conversionCode = cv::COLOR_BayerGB2BGR;
                    break;
                case PixelType_Gvsp_BayerRG8:
                    conversionCode = cv::COLOR_BayerRG2BGR;
                    break;
                case PixelType_Gvsp_BayerGR8:
                    conversionCode = cv::COLOR_BayerGR2BGR;
                    break;
                case PixelType_Gvsp_BayerBG8:
                    conversionCode = cv::COLOR_BayerBG2BGR;
                    break;
                default:
                    conversionCode = cv::COLOR_BayerGB2BGR; // 默认
            }
            
            cv::cvtColor(img, bgrImg, conversionCode);
            
            // 手动交换BGR通道
            std::vector<cv::Mat> channels(3);
            cv::split(bgrImg, channels);
            cv::Mat temp = channels[0];
            channels[0] = channels[2];
            channels[2] = temp;
            cv::merge(channels, bgrImg);
            
            outputImage = bgrImg;
            return true;
        }
        
        case PixelType_Gvsp_RGB8_Packed:  // RGB格式
        case PixelType_Gvsp_BGR8_Packed: {  // BGR格式
            cv::Mat img(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
            if (stImageInfo.enPixelType == PixelType_Gvsp_RGB8_Packed) {
                cv::cvtColor(img, outputImage, cv::COLOR_RGB2BGR);
            } else {
                outputImage = img;
            }
            return true;
        }
        
        case PixelType_Gvsp_Mono8:  // 黑白图像
            outputImage = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
            return true;
            
        default:
            std::cerr << "Unsupported pixel format: " << stImageInfo.enPixelType << std::endl;
            return false;
    }
}

// 枚举USB设备
std::vector<std::string> Camera::enumUSBDevices() {
    std::vector<std::string> deviceList;
    int nRet = MV_OK;
    
    // 初始化SDK（如果尚未初始化）
    static bool sdkInitialized = false;
    if (!sdkInitialized) {
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet) {
            std::cerr << "Initialize SDK fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return deviceList;
        }
        sdkInitialized = true;
    }
    
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet) {
        std::cerr << "Enum USB devices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return deviceList;
    }
    
    for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++) {
        MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
        if (pDeviceInfo->nTLayerType == MV_USB_DEVICE) {
            std::string deviceName = reinterpret_cast<char*>(pDeviceInfo->SpecialInfo.stUsb3VInfo.chModelName);
            std::string serialNumber = reinterpret_cast<char*>(pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            std::string deviceInfo = "Device " + std::to_string(i) + ": " + deviceName + " (SN: " + serialNumber + ")";
            deviceList.push_back(deviceInfo);
        }
    }
    
    return deviceList;
}
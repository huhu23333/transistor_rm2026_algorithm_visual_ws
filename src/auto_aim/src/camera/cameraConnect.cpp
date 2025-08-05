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

Camera::Camera(const std::string& deviceIp, const std::string& netIp) {
    int nRet = MV_OK;

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

    // 禁用自动曝光
    MV_CC_SetEnumValue(handle, "ExposureAuto", 0);
    std::cout << "Auto exposure disabled." << std::endl;

    // 禁用自动增益
    MV_CC_SetEnumValue(handle, "GainAuto", 0);
    std::cout << "Auto gain disabled." << std::endl;

    // 禁用白平衡自动
    MV_CC_SetEnumValue(handle, "BalanceWhiteAuto", 0);
    std::cout << "Auto white balance disabled." << std::endl;

    // 初始化相机参数
    if (!initCameraParams()) {
        std::cerr << "Failed to initialize camera parameters!" << std::endl;
        exit(1);
    }

    // 获取数据包大小
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    if (MV_OK != nRet) {
        std::cerr << "Get PayloadSize fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "Payload size: " << stParam.nCurValue << std::endl;
    }
    nPayloadSize = stParam.nCurValue;

    // 开始取流
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet) {
        std::cerr << "Start grabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        exit(1);
    } else {
        std::cout << "Grabbing started successfully." << std::endl;
    }

    // 启动取流线程
    pthread_t grabThread;
    pthread_create(&grabThread, NULL, workThread, this);
    std::cout << "Grabbing thread started." << std::endl;
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

            // 处理 BayerGB8 格式
            if (stImageInfo.enPixelType == PixelType_Gvsp_BayerGB8) {
                cv::Mat img(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
                cv::Mat bgrImg;
                cv::cvtColor(img, bgrImg, cv::COLOR_BayerGB2BGR);

                // 手动交换BGR通道
                std::vector<cv::Mat> channels(3);
                cv::split(bgrImg, channels);
                cv::Mat temp = channels[0];
                channels[0] = channels[2];
                channels[2] = temp;
                cv::merge(channels, bgrImg);

                // 线程锁定，更新图像
                pthread_mutex_lock(&g_mutex);
                g_image = bgrImg.clone();
                pthread_mutex_unlock(&g_mutex);
            } else {
                std::cerr << "Unsupported pixel format: " << stImageInfo.enPixelType << std::endl;
            }
        }

        if (g_bExit) {
            break;
        }
    }

    free(pData);
    std::cout << "Frame grabbing thread exiting." << std::endl;
    return NULL;
}
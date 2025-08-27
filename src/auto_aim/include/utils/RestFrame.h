// RestFrame.h
#ifndef REST_FRAME_H
#define REST_FRAME_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>


class RestFrame {
public:
    RestFrame(){}
    ~RestFrame(){}
    
    void updateCamOrientation(float yaw, float pitch, float roll);
    void updateCamPosition(float x, float y, float z);
    std::vector<float> getCamOrientation();
    std::vector<float> getCamPosition();
    std::vector<float> pnpResultToNormalFrame(float x_pnp, float y_pnp, float z_pnp);
    std::vector<float> getPositionInRestFrame(float x_cam, float y_cam, float z_cam);
    std::vector<float> getPositionInCamNormal(float x_global, float y_global, float z_global);
    std::vector<float> normalToPnpResultFrame(float x_cam_normal, float y_cam_normal, float z_cam_normal);

private:

    float camera_yaw;
    float camera_pitch;
    float camera_roll;
    float camera_x;
    float camera_y;
    float camera_z;

};




#endif // SHARED_MEMORY_TORCH_H
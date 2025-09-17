// RestFrame.h
#ifndef REST_FRAME_H
#define REST_FRAME_H

#include <vector>
#include <cmath>

class RestFrame {
public:
    RestFrame() : camera_yaw(0), camera_pitch(0), camera_roll(0), camera_x(0), camera_y(0), camera_z(0) {}
    ~RestFrame() {}
    
    void updateCamOrientation(float yaw, float pitch, float roll);
    void updateCamPosition(float x, float y, float z);
    std::vector<float> getCamOrientation();
    std::vector<float> getCamPosition();
    std::vector<float> pnpResultToNormalFrame(float x_pnp, float y_pnp, float z_pnp);
    std::vector<float> getWorldPositionFromCam(float x_cam_normal, float y_cam_normal, float z_cam_normal);
    std::vector<float> getCamPositionFromWorld(float x_global, float y_global, float z_global);
    std::vector<float> normalToPnpResultFrame(float x_cam_normal, float y_cam_normal, float z_cam_normal);

    std::vector<std::vector<float>> eulerToRotationMatrix(float yaw, float pitch, float roll);
    std::vector<std::vector<float>> multiplyRotationMatrixAndMatrix(
        const std::vector<std::vector<float>>& A, 
        const std::vector<std::vector<float>>& B);
    std::vector<float> getWorldEulerAnglesFromCam(float yaw_cam, float pitch_cam, float roll_cam);
    std::vector<float> getCamEulerAnglesFromWorld(float yaw_world, float pitch_world, float roll_world);

private:
    float camera_yaw;
    float camera_pitch;
    float camera_roll;
    float camera_x;
    float camera_y;
    float camera_z;
};

#endif // REST_FRAME_H
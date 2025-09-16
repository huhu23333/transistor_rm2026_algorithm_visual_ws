// RestFrame.cpp
#include "utils/RestFrame.h"

void RestFrame::updateCamOrientation(float yaw, float pitch, float roll) {
    camera_yaw = yaw;
    camera_pitch = pitch;
    camera_roll = roll;
}

void RestFrame::updateCamPosition(float x, float y, float z) {
    camera_x = x;
    camera_y = y;
    camera_z = z;
}

std::vector<float> RestFrame::getCamOrientation() {
    std::vector<float> result = {camera_yaw, camera_pitch, camera_roll};
    return result;
}

std::vector<float> RestFrame::getCamPosition() {
    std::vector<float> result = {camera_x, camera_y, camera_z};
    return result;
}

std::vector<float> RestFrame::pnpResultToNormalFrame(float x_pnp, float y_pnp, float z_pnp) { // 转为xyz向右向前向上
    std::vector<float> result = {x_pnp, z_pnp, -y_pnp};
    return result;
}

std::vector<float> RestFrame::getPositionInRestFrame(float x_cam_normal, float y_cam_normal, float z_cam_normal) {
    // roll
    float x_temp1 = x_cam_normal * std::cos(camera_roll) + z_cam_normal * std::sin(camera_roll);
    float y_temp1 = y_cam_normal;
    float z_temp1 = z_cam_normal * std::cos(camera_roll) - x_cam_normal * std::sin(camera_roll);
    // pitch
    float x_temp2 = x_temp1;
    float y_temp2 = y_temp1 * std::cos(camera_pitch) - z_temp1 * std::sin(camera_pitch);
    float z_temp2 = z_temp1 * std::cos(camera_pitch) + y_temp1 * std::sin(camera_pitch);
    // yaw
    float x_temp3 = x_temp2 * std::cos(camera_yaw) - y_temp2 * std::sin(camera_yaw);
    float y_temp3 = y_temp2 * std::cos(camera_yaw) + x_temp2 * std::sin(camera_yaw);
    float z_temp3 = z_temp2;
    // 平移
    float x_global = x_temp3 + camera_x;
    float y_global = y_temp3 + camera_y;
    float z_global = z_temp3 + camera_z;

    std::vector<float> result = {x_global, y_global, z_global};
    return result;
}

std::vector<float> RestFrame::getPositionInCamNormal(float x_global, float y_global, float z_global) {
    // 平移
    float x_temp1 = x_global - camera_x;
    float y_temp1 = y_global - camera_y;
    float z_temp1 = z_global - camera_z;
    // yaw
    float x_temp2 = x_temp1 * std::cos(-camera_yaw) - y_temp1 * std::sin(-camera_yaw);
    float y_temp2 = y_temp1 * std::cos(-camera_yaw) + x_temp1 * std::sin(-camera_yaw);
    float z_temp2 = z_temp1;
    // pitch
    float x_temp3 = x_temp2;
    float y_temp3 = y_temp2 * std::cos(-camera_pitch) - z_temp2 * std::sin(-camera_pitch);
    float z_temp3 = z_temp2 * std::cos(-camera_pitch) + y_temp2 * std::sin(-camera_pitch);
    // roll
    float x_cam_normal = x_temp3 * std::cos(-camera_roll) + z_temp3 * std::sin(-camera_roll);
    float y_cam_normal = y_temp3;
    float z_cam_normal = z_temp3 * std::cos(-camera_roll) - x_temp3 * std::sin(-camera_roll);
    
    std::vector<float> result = {x_cam_normal, y_cam_normal, z_cam_normal};
    return result;
}

std::vector<float> RestFrame::normalToPnpResultFrame(float x_cam_normal, float y_cam_normal, float z_cam_normal) {
    std::vector<float> result = {x_cam_normal, -z_cam_normal, y_cam_normal};
    return result;
}
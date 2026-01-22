#ifndef AUTO_AIM_MACRO_H
#define AUTO_AIM_MACRO_H

#define USE_VIDEO // 定义后使用视频而不是摄像头作为输入
// #define USE_IMAGES // 定义后使用图片而不是摄像头作为输入
// #define SAVE_IMG_FREQ 30 // 定义后将每n帧保存一次相机图片
// #define DEBUG_CODE // 定义后将在初始化结束后、装甲板识别代码前运行debug代码
#define SHOW_WINDOWS // 显示可视化窗口，使用自启动时注释掉
#define SYNC_CAMERA_FPS // 定义后主循环帧数将不会超过摄像头实际捕获帧数

#endif // AUTO_AIM_MACRO_H
#ifndef AUTO_AIM_MACRO_H
#define AUTO_AIM_MACRO_H

#define SAVE_IMG_FREQ 30 // 定义后将每n帧保存一次相机图片
#define SYNC_CAMERA_FPS // 定义后主循环帧数将不会超过摄像头实际捕获帧数

// #define FIX_ENEMY_COLOR 1 // 定义后将覆盖config.yaml及电控传来的敌方颜色，0为红色，1为蓝色
// #define FIX_BULLET_VELOCITY 12.0 // 定义后将覆盖config.yaml及电控传来的发弹初速度

// #define LOG_RESULT_VIDEO // 定义后将以视频形式记录处理结果
// #define LOG_ORIGIN_VIDEO // 定义后将记录处理的原始画面视频

// -!-!-!- 下面的宏定义在上场前都要注释掉再编译一遍；上面的视情况而定 -!-!-!-

#define USE_VIDEO // 定义后使用视频而不是摄像头作为输入
// #define USE_IMAGES // 定义后使用图片而不是摄像头作为输入
// #define DEBUG_CODE // 定义后将在初始化结束后、装甲板识别代码前运行debug代码
#define SHOW_WINDOWS // 显示可视化窗口，使用自启动时注释掉

// #define FILTER_ARMOR_CLASS 0b00000001  // 定义后将按照其的掩码值过滤掉指定类型的装甲板（具体类型参照ArmorType定义）（测试时有其他干扰装甲板使用这个）
// #define FIX_ARMOR_CLASS 2 // 定义后将会将所有未被过滤的类型的装甲板分类结果强制转换为指定类型（不会改变装甲板大小分类结果）（使用混装装甲板的目标车时使用这个)


#endif // AUTO_AIM_MACRO_H
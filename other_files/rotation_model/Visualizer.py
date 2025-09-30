import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow
import numpy as np
import time
import os
from matplotlib.animation import FFMpegWriter

class Visualizer:
    def __init__(self, save_video=False, video_filename="motion_visualization.mp4"):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.subplots_adjust(right=0.7)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Motion Model Visualization')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 视频保存相关属性
        self.save_video = save_video
        self.video_filename = video_filename
        self.writer = None
        self.frame_count = 0
        
        # 如果启用视频保存，初始化视频写入器
        if self.save_video:
            self._init_video_writer()
    
    def _init_video_writer(self):
        """初始化视频写入器，尝试不同的编码器设置"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.join(script_dir, self.video_filename)
            
            # 尝试不同的编码器参数组合
            metadata = dict(title='Motion Model Visualization', artist='Matplotlib')
            self.writer = FFMpegWriter(
                fps=30, 
                metadata=metadata, 
                codec='libx264',
                bitrate=2000,
                extra_args=['-pix_fmt', 'yuv420p']  # 添加像素格式参数
            )
            
            self.writer.setup(self.fig, video_path, dpi=100)
            print(f"开始录制视频: {video_path}")
            
        except Exception as e:
            print(f"初始化视频写入器失败: {e}")
            # 尝试备选方案
            try:
                self.writer = FFMpegWriter(fps=10)
                self.writer.setup(self.fig, video_path, dpi=100)
                print(f"使用简化参数开始录制视频: {video_path}")
            except Exception as e2:
                print(f"备选方案也失败: {e2}")
                self.save_video = False
                self.writer = None
    
    def update(self, true_params, fitted_params, observed_data):
        """更新可视化"""
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Motion Model Visualization')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 提取真实参数
        true_center_x = true_params['center_x']
        true_center_y = true_params['center_y']
        true_yaw = true_params['yaw']
        true_r = true_params['r']
        true_x = true_params['true_x']
        true_y = true_params['true_y']
        
        # 提取拟合参数
        fitted_center_x, fitted_vx, fitted_center_y, fitted_vy, fitted_center_z, fitted_vz, fitted_vyaw, fitted_r, fitted_all_armors_yaw = fitted_params
        
        fitted_yaw = fitted_all_armors_yaw[0]
        fitted_xs = fitted_center_x + fitted_r * np.sin(fitted_all_armors_yaw)
        fitted_ys = fitted_center_y - fitted_r * np.cos(fitted_all_armors_yaw)
        for i in range(len(fitted_all_armors_yaw)):
            fitted_x = fitted_xs[i]
            fitted_y = fitted_ys[i]
            self.ax.plot(true_x, true_y, 'g^', markersize=10)
            self.ax.plot(observed_data.x, observed_data.y, 'b*', markersize=12)
            self.ax.plot(fitted_x, fitted_y, 'rs', markersize=8)
            self.ax.plot([true_center_x, true_x], [true_center_y, true_y], 'g-', alpha=0.5)
            self.ax.plot([fitted_center_x, fitted_x], [fitted_center_y, fitted_y], 'r-', alpha=0.5)

        # 1. 绘制真实轨迹圆
        true_circle = patches.Circle((true_center_x, true_center_y), true_r, 
                                   fill=False, color='green', linestyle='-', linewidth=2, alpha=0.7)
        self.ax.add_patch(true_circle)
        
        # 2. 绘制拟合轨迹圆
        fitted_circle = patches.Circle((fitted_center_x, fitted_center_y), fitted_r, 
                                     fill=False, color='red', linestyle='--', linewidth=2, alpha=0.7)
        self.ax.add_patch(fitted_circle)
        
        # 3. 绘制真实中心点和朝向箭头
        self.ax.plot(true_center_x, true_center_y, 'go', markersize=8, label='True Center')
        arrow_length = true_r * 0.3
        true_arrow_dx = arrow_length * np.cos(true_yaw)
        true_arrow_dy = arrow_length * np.sin(true_yaw)
        self.ax.arrow(true_center_x, true_center_y, true_arrow_dx, true_arrow_dy, 
                     head_width=15, head_length=20, fc='green', ec='green', alpha=0.8)
        
        # 4. 绘制拟合中心点和朝向箭头
        self.ax.plot(fitted_center_x, fitted_center_y, 'ro', markersize=8, label='Fitted Center')
        fitted_arrow_dx = arrow_length * np.cos(fitted_yaw)
        fitted_arrow_dy = arrow_length * np.sin(fitted_yaw)
        self.ax.arrow(fitted_center_x, fitted_center_y, fitted_arrow_dx, fitted_arrow_dy, 
                     head_width=15, head_length=20, fc='red', ec='red', alpha=0.8)
        
        # 7. 设置坐标轴范围（动态调整）
        all_x = [true_center_x, true_x, fitted_center_x, *fitted_xs, observed_data.x]
        all_y = [true_center_y, true_y, fitted_center_y, *fitted_ys, observed_data.y]
        margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.3
        self.ax.set_xlim(min(min(all_x) - margin, -1000), max(max(all_x) + margin, 1000))
        self.ax.set_ylim(min(min(all_y) - margin, -1000), max(max(all_y) + margin, 3000))
        
        # 8. 添加图例
        self.ax.legend(loc='upper left')
        
        # 9. 在右侧添加参数文本
        text_x = 1500
        text_y = 500
        
        param_text = (
            f"True Parameters:\n"
            f"Center: ({true_center_x:.1f}, {true_center_y:.1f})\n"
            f"Yaw: {true_yaw:.3f} rad\n"
            f"Vyaw: {true_params['vyaw']:.3f} rad/s\n"
            f"Radius: {true_r:.1f}\n"
            f"True Pos: ({true_x:.1f}, {true_y:.1f})\n\n"
            f"Fitted Parameters:\n"
            f"Center: ({fitted_center_x:.1f}, {fitted_center_y:.1f})\n"
            f"Yaw: {fitted_yaw:.3f} rad\n"
            f"Vyaw: {fitted_vyaw:.3f} rad/s\n"
            f"Radius: {fitted_r:.1f}\n"
            f"Fitted Pos: ({fitted_x:.1f}, {fitted_y:.1f})\n\n"
            f"Observed:\n"
            f"Position: ({observed_data.x:.1f}, {observed_data.y:.1f})\n"
            f"Yaw: {observed_data.yaw:.3f} rad"
        )
        
        self.ax.text(text_x, text_y, param_text, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 关键修改：先绘制并暂停用于显示
        plt.draw()
        plt.pause(0.001)
        
        # 然后单独处理视频帧捕获
        if self.save_video and self.writer is not None:
            self._capture_frame()
    
    def _capture_frame(self):
        """专门处理帧捕获，确保渲染完成"""
        try:
            # 强制画布完成所有渲染
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # 添加小延迟确保完全渲染
            time.sleep(0.02)
            
            # 捕获帧
            self.writer.grab_frame()
            self.frame_count += 1
            
        except Exception as e:
            print(f"捕获帧时出错: {e}")
    
    def finish_video(self):
        """完成视频录制并保存"""
        if self.save_video and self.writer is not None:
            try:
                # 添加最终延迟确保最后一帧被正确捕获
                time.sleep(0.1)
                self.writer.finish()
                print(f"视频保存完成！共保存 {self.frame_count} 帧")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                video_path = os.path.join(script_dir, self.video_filename)
                print(f"视频文件保存在: {video_path}")
            except Exception as e:
                print(f"保存视频时出错: {e}")
            finally:
                self.writer = None
                self.save_video = False
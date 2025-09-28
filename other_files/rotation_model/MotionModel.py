import numpy as np
from scipy.optimize import least_squares
import math
import time
np.random.seed(42)



class ObservedData:
    def __init__(self, x, y, z, yaw, t):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.t = t


def center_to_yaw_function(params, armor_x, armor_y, data_t):
    center_x = params[0]
    center_vx = params[1]
    center_y = params[2]
    center_vy = params[3]
    center_yaw = np.arctan2(-(center_x + data_t * center_vx - armor_x), (center_y + data_t * center_vy - armor_y))
    return center_yaw

def center_residuals(params, armor_yaw, armor_x, armor_y, armor_z, data_t):
    center_yaw = center_to_yaw_function(params, armor_x, armor_y, data_t)
    center_z = params[4]
    center_vz = params[5]
    res_yaw = center_yaw - armor_yaw 
    res_z = center_z + data_t * center_vz - armor_z
    return np.concatenate([res_yaw, res_z])

class MotionModel:
    def __init__(self, init_observed_data):
        self.observed_data_history = [init_observed_data]
        self.last_observed_data = init_observed_data
        self.last_x = init_observed_data.x
        self.last_y = init_observed_data.y
        self.last_z = init_observed_data.z
        self.last_yaw = init_observed_data.yaw
        self.last_t = init_observed_data.t
        self.center_vx = 0.0
        self.center_vy = 0.0
        self.center_vz = 0.0
        self.vyaw = 0.0
        self.r = 250
        self.center_x = self.last_x - self.r * math.sin(self.last_yaw)
        self.center_y = self.last_y + self.r * math.cos(self.last_yaw)
        self.center_z = self.last_z
        self.inited_v = False
        self.max_history = 90

        # 新增属性用于旋转参数拟合
        self.rotation_period = 0.0
        self.current_phase = 0.0
        self.n_armors = 3  # 假设3等分圆周
        self.freq_refine_multiple = 10

        self.jump_delta_yaw = math.pi / 6
        self.jump_count = 0
        self.jump_rad = math.pi * 2 / self.n_armors

        self.jump_to_mid_delta_phase = math.pi / 3 - math.pi * 5 / 180

        self.all_armors_yaw = np.array([self.last_yaw] * self.n_armors)

    def update(self, observed_data):
        self.observed_data_history.append(observed_data)
        if len(self.observed_data_history) > self.max_history:
            self.observed_data_history = self.observed_data_history[len(self.observed_data_history) - self.max_history:]
        initial_params, t_data, x_data, y_data, z_data, yaw_data = self.get_params()
        center_fit_result = least_squares(center_residuals, np.array([self.center_x, self.center_vx, self.center_y, self.center_vy, self.center_z, self.center_vz]), args=(yaw_data, x_data, y_data, z_data, t_data))
        center_fit_params = center_fit_result.x
        self.center_x = center_fit_params[0]
        self.center_y = center_fit_params[2]
        self.center_z = center_fit_params[4]
        self.center_vx = center_fit_params[1]
        self.center_vy = center_fit_params[3]
        self.center_vz = center_fit_params[5]

        self.calculate_r()
        
        # 新增：拟合旋转参数
        self.fit_rotation_parameters()

        _, _, _, now_armor_yaw = self.predict(0)
        self.last_yaw = now_armor_yaw
        self.all_armors_yaw = np.array([now_armor_yaw + i * self.jump_rad for i in range(self.n_armors)])

    def calculate_r(self):
        initial_params, t_data, x_data, y_data, z_data, yaw_data = self.get_params()
        self.r =  np.mean(np.sqrt((x_data - (self.center_x + t_data * self.center_vx)) ** 2 + (y_data - (self.center_y + t_data * self.center_vy)) ** 2))

    def get_params(self):
        params = np.array([self.center_x, self.center_vx, self.center_y, self.center_vy, self.center_z, self.center_vz, self.last_yaw, self.vyaw, self.r])
        t_data = np.array([observed_data.t-self.observed_data_history[-1].t for observed_data in self.observed_data_history])
        x_data = np.array([observed_data.x for observed_data in self.observed_data_history])
        y_data = np.array([observed_data.y for observed_data in self.observed_data_history])
        z_data = np.array([observed_data.z for observed_data in self.observed_data_history])
        yaw_data = np.array([observed_data.yaw for observed_data in self.observed_data_history])
        return [params, t_data, x_data, y_data, z_data, yaw_data]
    
    def interpolate_data(self, data, pos):
        """线性插值函数"""
        if pos < 0 or pos >= len(data) - 1:
            return data[int(pos)] if 0 <= int(pos) < len(data) else 0.0
        
        idx_low = int(pos)
        idx_high = idx_low + 1
        weight_high = pos - idx_low
        weight_low = 1.0 - weight_high
        
        return weight_low * data[idx_low] + weight_high * data[idx_high]

    def compute_modified_acf(self, residual):
        """计算修改后的自相关函数（支持细化步长）"""
        n = len(residual)
        if n == 0:
            return []
        
        residual_mean = np.mean(residual)
        max_lag = int(n * 0.8)
        
        # 细化后的lag点数
        refined_max_lag = max_lag * self.freq_refine_multiple
        modified_acf = np.zeros(refined_max_lag + 1)
        
        for k_idx in range(refined_max_lag + 1):
            k = k_idx / self.freq_refine_multiple  # 细化后的lag值
            
            if k == 0:
                # k=0时的特殊情况处理
                numerator = np.sum((residual - residual_mean) ** 2) / n
            else:
                numerator = 0.0
                count = 0
                
                # 使用插值计算非整数lag的ACF
                for t in range(n - int(k) - 1):
                    # 对residual[t+k]进行插值
                    interp_pos = t + k
                    if interp_pos < n:
                        interp_value = self.interpolate_data(residual, interp_pos)
                        numerator += (residual[t] - residual_mean) * (interp_value - residual_mean)
                        count += 1
                
                if count > 0:
                    numerator /= count
            
            modified_acf[k_idx] = numerator
        
        return modified_acf

    def find_period(self, modified_acf):
        """从修改后的ACF中寻找周期（支持细化步长）"""
        if len(modified_acf) < 2:
            return 1
        
        max_k_idx = 1
        max_value = modified_acf[1]
        last_modified_acf = modified_acf[1]
        
        # 寻找第一个下降点（使用细化步长）
        search_range = min(int(len(modified_acf) / 2), len(modified_acf))
        for k_idx in range(2, search_range):
            if modified_acf[k_idx] < 0:
                max_k_idx = k_idx
                max_value = modified_acf[k_idx]
                last_modified_acf = modified_acf[k_idx]
                break
        
        modified_acf_updating = False
        for k_idx in range(max_k_idx + 1, len(modified_acf)):
            if (modified_acf[k_idx] > max_value * 3.0) or (modified_acf_updating and modified_acf[k_idx] > max_value * 0.8):
                if modified_acf[k_idx] > last_modified_acf:
                    modified_acf_updating = True
                if modified_acf[k_idx] > max_value:
                    max_value = modified_acf[k_idx]
                    max_k_idx = k_idx
            if modified_acf[k_idx] < last_modified_acf * 0.8:
                modified_acf_updating = False
            last_modified_acf = modified_acf[k_idx]
        
        # 将细化索引转换为实际周期
        return max_k_idx / self.freq_refine_multiple

    def compute_change_intensity(self, data):
        """计算变化强度（支持细化步长）"""
        n = len(data)
        if n < 2:
            return np.zeros(n * self.freq_refine_multiple)
        
        # 先计算整数点的变化强度
        intensity_integer = np.zeros(n)
        for i in range(1, n):
            intensity_integer[i] = (data[i] - data[i-1]) ** 2
        
        # 使用线性插值得到细化后的变化强度
        intensity_refined = np.zeros(n * self.freq_refine_multiple)
        for i in range(len(intensity_refined)):
            pos = i / self.freq_refine_multiple
            intensity_refined[i] = self.interpolate_data(intensity_integer, pos)
        
        return intensity_refined

    def smooth_data(self, data, window_size=5):
        """使用滑动窗口平滑数据"""
        if len(data) < window_size:
            return data
        
        smoothed = np.zeros(len(data))
        half_window = window_size // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            smoothed[i] = np.mean(data[start:end])
        
        return smoothed

    def find_jump_points(self, intensity, threshold_factor=2.0):
        """寻找跳变点（使用细化步长）"""
        if len(intensity) == 0:
            return []
        
        mean_intensity = np.mean(intensity)
        std_intensity = np.std(intensity)
        threshold = mean_intensity + threshold_factor * std_intensity
        
        jump_points = []
        for i in range(1, len(intensity)):
            if intensity[i] > threshold and intensity[i] > intensity[i-1]:
                # 将细化索引转换为原始索引
                jump_points.append(i / self.freq_refine_multiple)
        
        return jump_points

    def fit_rotation_parameters(self):
        """拟合旋转参数：旋转速度、半径和当前角度"""
        if len(self.observed_data_history) < 10:
            return
        
        # 提取观测数据
        t_data = np.array([obs.t for obs in self.observed_data_history])
        x_data = np.array([obs.x for obs in self.observed_data_history])
        y_data = np.array([obs.y for obs in self.observed_data_history])
        z_data = np.array([obs.z for obs in self.observed_data_history])
        yaw_data = np.array([obs.yaw for obs in self.observed_data_history])
        
        # 计算四个分量的ACF（使用细化步长）
        acf_x = self.compute_modified_acf(x_data)
        acf_y = self.compute_modified_acf(y_data)
        acf_z = self.compute_modified_acf(z_data)
        acf_yaw = self.compute_modified_acf(yaw_data)
        
        # 合并ACF（简单相加）
        combined_acf = np.zeros(len(acf_x))
        for i in range(len(acf_x)):
            combined_acf[i] = acf_x[i] + acf_y[i] + acf_z[i] + acf_yaw[i]
        
        # 寻找周期（使用细化步长）
        period_frames = self.find_period(combined_acf)
        
        if period_frames > 1 and period_frames < len(t_data) / 2:
            # 计算时间间隔
            time_intervals = np.diff(t_data)
            avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 0.01
            
            # 计算旋转周期
            self.rotation_period = period_frames * avg_interval * self.n_armors
            self.vyaw = 2 * math.pi / self.rotation_period if self.rotation_period > 0 else 0.0
        
        # 使用变化强度检测跳变点（使用细化步长）
        intensity_x = self.compute_change_intensity(x_data)
        intensity_y = self.compute_change_intensity(y_data)
        intensity_z = self.compute_change_intensity(z_data)
        intensity_yaw = self.compute_change_intensity(yaw_data)
        
        # 合并变化强度
        combined_intensity = intensity_x + intensity_y + intensity_z + intensity_yaw
        smoothed_intensity = self.smooth_data(combined_intensity)
        
        # 寻找跳变点
        jump_points = self.find_jump_points(smoothed_intensity)
        
        if len(jump_points) > 0:
            # 使用最近的跳变点确定相位
            last_jump_idx = jump_points[-1]
            if last_jump_idx < len(t_data):
                time_since_jump = t_data[-1] - t_data[int(last_jump_idx)]
                self.current_phase = (time_since_jump * self.vyaw - self.jump_to_mid_delta_phase) % (2 * math.pi)

    def predict(self, predict_time):
        """预测未来时刻的位置和角度"""
        delta_t = predict_time - self.last_t
        
        # 预测中心点位置
        pred_center_x = self.center_x + delta_t * self.center_vx
        pred_center_y = self.center_y + delta_t * self.center_vy
        pred_center_z = self.center_z + delta_t * self.center_vz
        
        # 预测角度
        pred_phase = (self.current_phase + delta_t * self.vyaw) % (2 * math.pi)
        
        # 预测装甲板位置
        pred_x = pred_center_x + self.r * math.sin(pred_phase)
        pred_y = pred_center_y - self.r * math.cos(pred_phase)
        pred_z = pred_center_z
        pred_yaw = pred_phase
            
        return pred_x, pred_y, pred_z, pred_yaw
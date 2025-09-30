import numpy as np
from scipy.optimize import least_squares
import math
import time
np.random.seed(42)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.fftpack import fft, ifft

class ObservedData:
    def __init__(self, x, y, z, yaw, t):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.t = t

# todo:
"""
使用最小二乘法解析解计算z vz
使用矩阵运算拟合x vx y vy
中心拟合离轴向量权重衰减
"""

def center_residuals(params, armor_yaw, armor_x, armor_y, armor_z, data_t):
    center_x = params[0]
    center_vx = params[1]
    center_y = params[2]
    center_vy = params[3]
    center_z = params[4]
    center_vz = params[5]
    axis_vector = np.stack([-np.sin(armor_yaw), np.cos(armor_yaw)])
    off_axis_vector = np.stack([axis_vector[1], -axis_vector[0]])
    aromr_to_center_vector = np.stack([center_x + data_t * center_vx - armor_x, center_y + data_t * center_vy - armor_y])
    res_xy = aromr_to_center_vector[0] * off_axis_vector[0] + aromr_to_center_vector[1] * off_axis_vector[1]
    res_z = center_z + data_t * center_vz - armor_z
    return np.concatenate([res_xy, res_z])


def compute_modified_acf(residual):
    n = len(residual)
    if n == 0:
        return []
    residual_mean = np.mean(residual)
    max_lag = int(n * 0.8)
    modified_acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            modified_acf[k] = np.sum((residual - residual_mean) ** 2) / len(residual)
        else:
            modified_acf[k] = np.sum((residual[:len(residual)-k] - residual_mean) * 
                                            (residual[k:] - residual_mean)
                                            ) / (len(residual) - k)
    return modified_acf

def variance(signal):
    signal_mean = np.mean(signal)
    return np.sum((signal - signal_mean) ** 2) / len(signal)

def linear_interpolation(data, refine_multiple):
    refine_multiple = int(refine_multiple)
    result_len = (len(data) - 1) * refine_multiple + 1
    result = np.zeros(result_len)
    for result_i in range(result_len):
        origin_i = result_i // refine_multiple
        result_i_left_part = result_i - origin_i * refine_multiple
        if result_i_left_part == 0:
            result[result_i] = data[origin_i]
        else:
            weight_high = result_i_left_part / refine_multiple
            weight_low = 1 - weight_high
            result[result_i] = weight_low * data[origin_i] + weight_high * data[origin_i + 1]
    return result  

def lag_stack_with_decay(signal, refine_multiple = 1):
    refined_signal = linear_interpolation(signal, refine_multiple)
    result_len = len(refined_signal)
    result = np.zeros(result_len)
    for lag in range(1, result_len):
        lag_n = result_len // lag
        lag_left = result_len - lag_n * lag
        temp = np.zeros(lag)
        for lag_i in range(lag_n):
            temp += refined_signal[lag_i*lag : (lag_i+1)*lag]
        if lag_left > 0:
            temp[:lag_left] += refined_signal[-lag_left:]
            temp[:lag_left] /= (lag_n + 1)
            temp[lag_left:] /= lag_n
        else:
            temp /= lag_n
        result[lag] = variance(temp) / lag
    return result


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
        self.max_history = 60

        self.refine_multiple = 10

        # 新增属性用于旋转参数拟合
        self.rotation_period = 0.0
        self.current_phase = 0.0
        self.n_armors = 4  # 假设3等分圆周

        self.jump_rad = math.pi * 2 / self.n_armors
        self.delta_phase = 0 #math.pi * 15 / 180

        self.all_armors_yaw = np.array([self.last_yaw] * self.n_armors)
        
        self.debug_fig, self.debug_ax = plt.subplots(2, 2, figsize=(8, 8))

        self.rotation_direction = 1


    def update(self, observed_data):
        self.observed_data_history.append(observed_data)
        if len(self.observed_data_history) > self.max_history:
            self.observed_data_history = self.observed_data_history[len(self.observed_data_history) - self.max_history:]
        initial_params, t_data, x_data, y_data, z_data, yaw_data = self.get_params()
        n_latest = 10
        max_r = 800
        max_v = 5000
        bounds = (np.array([np.mean(x_data[-n_latest:])-max_r, -max_v, np.mean(y_data[-n_latest:])-max_r, -max_v, -np.inf, -np.inf]), 
                  np.array([np.mean(x_data[-n_latest:])+max_r, max_v, np.mean(y_data[-n_latest:])+max_r, max_v, np.inf, np.inf]))
        init_in_range = np.clip(np.array([self.center_x, self.center_vx, self.center_y, self.center_vy, self.center_z, self.center_vz]), 
                                *bounds)
        #print(init_in_range)
        center_fit_result = least_squares(center_residuals, 
                                          init_in_range, 
                                          args=(yaw_data, x_data, y_data, z_data, t_data), 
                                          bounds=bounds)
        if center_fit_result.success:
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
    
    def find_mid_yaw(self, yaw_data, period_frames):
        mid_square_yaw_data = (yaw_data - np.mean(yaw_data)) ** 2
        data_to_fit = np.tanh(mid_square_yaw_data * 3)

        n = len(data_to_fit)
        ts = np.linspace(0, n-1, n)
        thetas = ts / period_frames * 2 * np.pi
        a0 = np.sum(data_to_fit) / n
        a1 = np.sum(data_to_fit * np.cos(thetas)) * 2 / n
        b1 = np.sum(data_to_fit * np.sin(thetas)) * 2 / n

        phi = np.arctan2(b1, a1)
        A = np.sqrt(a1 **2 + b1 ** 2)
        thetas_fitted = thetas - phi
        fitted_data = a0 + A * np.cos(thetas_fitted)

        mid_points = []
        for i in range(1, n-1):
            if fitted_data[i] <= fitted_data[i-1] and fitted_data[i] <= fitted_data[i+1]:
                mid_points.append(i)

        return mid_points, data_to_fit, fitted_data
    
    def get_rotation_direction(self, mid_points, yaw_data):
        d_yaw_integrate = 0
        if len(yaw_data) < 2:
            return 1
        for mid_point_idx in mid_points:
            if mid_point_idx == 0:
                d_yaw_integrate += yaw_data[1] - yaw_data[0]
            elif mid_point_idx == len(yaw_data) - 1:
                d_yaw_integrate += yaw_data[-1] - yaw_data[-2]
            else:
                d_yaw_integrate += yaw_data[mid_point_idx+1] - yaw_data[mid_point_idx-1]
        if d_yaw_integrate > 0:
            return 1
        else:
            return -1

    def fit_rotation_parameters(self):
        """拟合旋转参数：旋转速度和当前角度"""
        if len(self.observed_data_history) < 10:
            return
        
        # 提取观测数据
        t_data = np.array([obs.t for obs in self.observed_data_history])
        x_data = np.array([obs.x for obs in self.observed_data_history]) - self.center_vx * t_data
        y_data = np.array([obs.y for obs in self.observed_data_history]) - self.center_vy * t_data
        z_data = np.array([obs.z for obs in self.observed_data_history]) - self.center_vz * t_data
        yaw_data = np.array([obs.yaw for obs in self.observed_data_history])
        
        # 计算四个分量的ACF
        acf_x = compute_modified_acf(x_data)
        acf_y = compute_modified_acf(y_data)
        acf_z = compute_modified_acf(z_data)
        acf_yaw = compute_modified_acf(yaw_data)
        
        # 合并ACF（简单相加）
        combined_acf = acf_x + acf_y + acf_z + acf_yaw

        refined_acf_lag_stack_with_decay = lag_stack_with_decay(combined_acf, self.refine_multiple)
        period_frames = refined_acf_lag_stack_with_decay.argmax() / self.refine_multiple
        
        if period_frames > 1 and period_frames < len(t_data) / 2:
            # 计算时间间隔
            time_intervals = np.diff(t_data)
            avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 0.01
            
            # 计算旋转周期
            self.rotation_period = period_frames * avg_interval * self.n_armors
            self.vyaw = 2 * math.pi / self.rotation_period if self.rotation_period > 0 else 0.0

        mid_points, data_to_fit, fitted_mid_data = self.find_mid_yaw(yaw_data, period_frames)
        self.rotation_direction = self.get_rotation_direction(mid_points, yaw_data)
        self.vyaw *= self.rotation_direction
        if len(mid_points) > 0:
            last_mid_idx = mid_points[-1]
            if last_mid_idx < len(t_data):
                time_since_mid = t_data[-1] - t_data[int(last_mid_idx)]
                self.current_phase = (time_since_mid * self.vyaw + self.delta_phase * self.rotation_direction) % (2 * math.pi)

        self.debug_ax[0][0].clear()
        self.debug_ax[0][0].plot(np.linspace(0,len(x_data)-1,len(x_data)),x_data, label='x')
        self.debug_ax[0][0].plot(np.linspace(0,len(y_data)-1,len(y_data)),y_data, label='y')
        self.debug_ax[0][0].plot(np.linspace(0,len(z_data)-1,len(z_data)),z_data, label='z')
        self.debug_ax[0][0].plot(np.linspace(0,len(yaw_data)-1,len(yaw_data)),yaw_data*200, label='yaw * 200')
        for mid_point_idx in mid_points:
            self.debug_ax[0][0].plot(mid_point_idx, 0, 'g^', markersize=10)
        self.debug_ax[0][0].legend(loc='upper left')

        self.debug_ax[0][1].clear()
        self.debug_ax[0][1].plot(np.linspace(0,len(acf_x)-1,len(acf_x)),acf_x, label='acf_x')
        self.debug_ax[0][1].plot(np.linspace(0,len(acf_y)-1,len(acf_y)),acf_y, label='acf_y')
        self.debug_ax[0][1].plot(np.linspace(0,len(acf_z)-1,len(acf_z)),acf_z, label='acf_z')
        self.debug_ax[0][1].plot(np.linspace(0,len(acf_yaw)-1,len(acf_yaw)),acf_yaw, label='acf_yaw')
        self.debug_ax[0][1].plot(np.linspace(0,len(combined_acf)-1,len(combined_acf)),combined_acf, label='combined_acf')
        self.debug_ax[0][1].legend(loc='upper right')

        self.debug_ax[1][0].clear()
        self.debug_ax[1][0].plot(np.linspace(0,len(refined_acf_lag_stack_with_decay)/self.refine_multiple-1,len(refined_acf_lag_stack_with_decay)),
                                 refined_acf_lag_stack_with_decay, label='refined_acf_lag_stack_with_decay')
        self.debug_ax[1][0].plot(period_frames, refined_acf_lag_stack_with_decay[int(period_frames * self.refine_multiple)], 'g^', markersize=10)
        self.debug_ax[1][0].legend(loc='upper left')

        self.debug_ax[1][1].clear()
        self.debug_ax[1][1].plot(np.linspace(0,len(data_to_fit)-1,len(data_to_fit)),data_to_fit, label='data_to_fit')
        self.debug_ax[1][1].plot(np.linspace(0,len(fitted_mid_data)-1,len(fitted_mid_data)),fitted_mid_data, label='fitted_mid_data')
        for mid_point_idx in mid_points:
            self.debug_ax[1][1].plot(mid_point_idx, 0, 'g^', markersize=10)
        self.debug_ax[1][1].legend(loc='upper right')


        self.debug_fig.canvas.draw()
        self.debug_fig.canvas.flush_events()

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
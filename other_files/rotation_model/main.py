import os,sys
sys.path.append(os.path.dirname(__file__))
import time
import matplotlib.pyplot as plt

import FakeEnv
import MotionModel
import Visualizer

def main():
    env = FakeEnv.FakeEnv()
    visualizer = Visualizer.Visualizer(save_video=True)
    
    init_obs_data = MotionModel.ObservedData(*env.observeDataWithNoise(0),0)
    motion_model = MotionModel.MotionModel(init_obs_data)
    last_t = time.time()
    
    # 添加时间参考点
    init_time = time.time()
    
    print("Starting visualization... Press Ctrl+C to stop.")
    
    try:
        t = 0
        while True:
            # 获取真实参数（无噪音）
            t += 0.033
            true_params = env.get_true_params(t)
            true_params['current_time_ref'] = init_time
            
            # 获取观测数据（有噪音）
            obs_data = MotionModel.ObservedData(*env.observeDataWithNoise(t),t)
            if abs(obs_data.yaw) > 0.7:
                pred_data = motion_model.predict(0.033)
                obs_data = MotionModel.ObservedData(*pred_data, t)
                motion_model.update(obs_data)
            else:
                motion_model.update(obs_data)
            
            # 获取拟合参数
            fitted_params = [motion_model.center_x, motion_model.center_vx, 
                           motion_model.center_y, motion_model.center_vy,
                           motion_model.center_z, motion_model.center_vz,
                           motion_model.vyaw, 
                           motion_model.r, motion_model.all_armors_yaw]
            
            # 更新可视化
            visualizer.update(true_params, fitted_params, obs_data)
            
            # 控制帧率
            while time.time() - last_t < 0.033:  # 30 FPS for better visualization
                time.sleep(0.001)
            last_t = time.time()
            
    except KeyboardInterrupt:
        print("\nStopped by user")
        visualizer.finish_video()
    """ except Exception as e:
        print(f"Error: {e}") """
    
    print("Visualization finished.")
    #plt.show(block=True)  # 保持窗口打开

if __name__ == "__main__":
    main()
// std
#include<functional>
// Eigen
#include<Eigen/Dense>
// ceres
#include <ceres/jet.h>

// --- 新增头文件 ---
#include "rclcpp/rclcpp.hpp"
#include <sstream>
// --- 新增头文件结束 ---

/*
扩展卡尔曼滤波器 (EKF)
*/

template <int N_x,            // 状态向量维度
          int N_z,            // 测量向量维度
          class PredicFunc,   // 过程模型函数类型
          class MeasureFunc>  // 测量模型函数类型
class ExtendedKalmanFilter{
public:
  ExtendedKalmanFilter() = default;   // 默认构造函数

  using MatrixXX = Eigen::Matrix<double, N_x, N_x>; // 状态转移矩阵
  using MatrixXZ = Eigen::Matrix<double, N_x, N_z>; // 卡尔曼增益矩阵
  using MatrixZX = Eigen::Matrix<double, N_z, N_x>; // 雅可比矩阵
  using MatrixZZ = Eigen::Matrix<double, N_z, N_z>; // 测量噪声协方差矩阵
  using MatrixX1 = Eigen::Matrix<double, N_x, 1>;   // 状态向量
  using MatrixZ1 = Eigen::Matrix<double, N_z, 1>;   // 测量向量

  using UpdateQFunc = std::function<MatrixXX()>;
  using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1 &z)>;

  // 构造函数
  explicit ExtendedKalmanFilter(const PredicFunc &f,          // 过程模型函数
                                const MeasureFunc &h,         // 测量模型
                                const UpdateQFunc &updateQ,   // 过程噪声协方差更新函数
                                const UpdateRFunc &updateR,   // 测量噪声协方差更新函数
                                const MatrixXX &P0) noexcept  // 先验估计协方差
      : f(f), h(h), update_Q(updateQ), update_R(updateR), P_post(P0) {
        F = MatrixXX::Zero();   //过程模型的雅可比矩阵
        H = MatrixZX::Zero();   //测量模型的雅可比矩阵
  }

  // 状态初始化, 设置初始状态的函数，将状态向量 x_post 设置为初始值 x0
  void setState(const MatrixX1 &x0) noexcept { x_post = x0; }

  // 设置过程模型的函数
  void setPredictFunc(const PredicFunc  &f) noexcept { this->f = f; }

  // 设置测量模型的函数
  void setMeasureFunc(const MeasureFunc &h) noexcept { this->h = h; }

  // 
  MatrixX1 predict() noexcept {
      ceres::Jet<double, N_x> x_e_jet[N_x];
      for (int i = 0; i < N_x; ++i) {
        x_e_jet[i].a = x_post[i];
        //x_e_jet[i].v.setZero();
        x_e_jet[i].v[i] = 1.0;
      }

      // 调用过程模型
      ceres::Jet<double, N_x> x_p_jet[N_x];
      f(x_e_jet, x_p_jet);
      for (int i = 0; i < N_x; ++i) {
        x_pri[i] = x_p_jet[i].a;
        // 关键修正：在模板类中调用 block<...> 需要 template 关键字
        F.block(i, 0, 1, N_x) = x_p_jet[i].v.transpose();
      }

      Q = update_Q();
      P_pri = F * P_post * F.transpose() + Q;
      x_post = x_pri;

      return x_pri;
  }
  MatrixX1 getState() const noexcept { return x_post; }
  //
  MatrixX1 update(const MatrixZ1 &z) noexcept {
      ceres::Jet<double, N_x> x_p_jet[N_x];
      for (int i = 0; i < N_x; i++) {
        x_p_jet[i].a = x_pri[i];
        x_p_jet[i].v[i] = 1;
      }
      ceres::Jet<double, N_x> z_p_jet[N_z];
      h(x_p_jet, z_p_jet);

      MatrixZ1 z_pri;
      for (int i = 0; i < N_z; i++) {
        z_pri[i] = z_p_jet[i].a;
        H.block(i, 0, 1, N_x) = z_p_jet[i].v.transpose();
      }

      R = update_R(z);
      // ======================= 新增的ROS2日志调试代码 START =======================
      // 获取一个全局的logger实例来进行打印
      auto logger = rclcpp::get_logger("ekf_debug_logger");

      // 创建一个字符串流来格式化矩阵
      std::stringstream ss_p, ss_h, ss_r, ss_s;

      ss_p << P_pri;
      // RCLCPP_DEBUG(logger, "----------- EKF DEBUG INFO (Pre-Inverse) -----------");
      // RCLCPP_DEBUG(logger, "P_pri (Predicted Covariance):\n%s", ss_p.str().c_str());

      ss_h << H;
      // RCLCPP_DEBUG(logger, "H (Measurement Jacobian):\n%s", ss_h.str().c_str());

      ss_r << R;
      // RCLCPP_DEBUG(logger, "R (Measurement Noise):\n%s", ss_r.str().c_str());

      // 计算并打印即将被求逆的矩阵 S
      MatrixZZ S = H * P_pri * H.transpose() + R;
      ss_s << S;
      // RCLCPP_DEBUG(logger, "S (Matrix to be inverted):\n%s", ss_s.str().c_str());

      // 计算并打印 S 的行列式
      double detS = S.determinant();
      // RCLCPP_DEBUG(logger, "Determinant of S: %e", detS); // 使用 %e 科学计数法打印
      
      if (std::abs(detS) < 1e-9) {
          RCLCPP_ERROR(logger, "CRITICAL: Determinant is close to zero! Matrix inversion will fail.");
      }
      // RCLCPP_DEBUG(logger, "----------------------------------------------------");
      // ======================= 新增的ROS2日志调试代码 END =======================

      
      // 计算卡尔曼增益
      K = P_pri * H.transpose() * (H * P_pri * H.transpose() + R).inverse();
      x_post = x_post + K * (z - z_pri);
      P_post = (MatrixXX::Identity() - K * H) * P_pri;
    
      return x_post;
  }


private:
  // 过程非线性向量函数
  PredicFunc f;
  MatrixXX F;
  // 观测非线性向量函数
  MeasureFunc h;
  MatrixZX H;
  // 过程噪声协方差矩阵
  UpdateQFunc update_Q;
  MatrixXX Q;
  // 测量噪声协方差矩阵
  UpdateRFunc update_R;
  MatrixZZ R;

  // 先验误差估计协方差矩阵
  MatrixXX P_pri;
  // 后验误差估计协方差矩阵
  MatrixXX P_post;

  // 卡尔曼增益
  MatrixXZ K;

  // 先验状态
  MatrixX1 x_pri;
  // 后验状态
  MatrixX1 x_post;

};
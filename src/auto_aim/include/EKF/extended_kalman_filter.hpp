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

      P_pri = 0.5 * (P_pri + P_pri.transpose()).eval();
      P_post = 0.5 * (P_post + P_post.transpose()).eval();

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
      // ======================= ROS2 日志调试代码 START =======================
      // 获取一个全局的logger实例来进行打印
      auto logger = rclcpp::get_logger("ekf_debug_logger");

      // 创建一个字符串流来格式化矩阵
      std::stringstream ss_z, ss_x_pri, ss_F, ss_Q, ss_z_pri, ss_p_pri, ss_h, ss_r, ss_s, ss_k, ss_p_post, ss_x_post;
      RCLCPP_INFO(logger, "----------- EKF DEBUG INFO -----------");

      ss_z << z;
      RCLCPP_INFO(logger, "z:\n%s", ss_z.str().c_str());

      ss_x_pri << x_pri;
      RCLCPP_INFO(logger, "x_pri:\n%s", ss_x_pri.str().c_str());

      ss_F << F;
      RCLCPP_INFO(logger, "F:\n%s", ss_F.str().c_str());

      ss_Q << Q;
      RCLCPP_INFO(logger, "Q:\n%s", ss_Q.str().c_str());

      ss_z_pri << z_pri;
      RCLCPP_INFO(logger, "z_pri:\n%s", ss_z_pri.str().c_str());

      ss_p_pri << P_pri;
      RCLCPP_INFO(logger, "P_pri (Predicted Covariance):\n%s", ss_p_pri.str().c_str());

      ss_h << H;
      RCLCPP_INFO(logger, "H (Measurement Jacobian):\n%s", ss_h.str().c_str());

      ss_r << R;
      RCLCPP_INFO(logger, "R (Measurement Noise):\n%s", ss_r.str().c_str());

      // 计算并打印即将被求逆的矩阵 S
      MatrixZZ S = H * P_pri * H.transpose() + R;
      ss_s << S;
      RCLCPP_INFO(logger, "S (Innovation Covariance):\n%s", ss_s.str().c_str());

      // 计算并打印 S 的行列式
      double detS = S.determinant();
      RCLCPP_INFO(logger, "Determinant of S: %e", detS); // 使用 %e 科学计数法打印
      
      if (std::abs(detS) < 1e-9) {
          RCLCPP_ERROR(logger, "CRITICAL: Determinant is close to zero! Matrix inversion will fail.");
      }
      // ======================= ROS2 日志调试代码 END =======================

      

      // 计算卡尔曼增益// 替换这行:
      // K = P_pri * H.transpose() * S.inverse();
      // 使用SVD分解求解 K = P_pri * H.transpose() * S^{-1}
      Eigen::JacobiSVD<MatrixZZ> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
      // 获取奇异值
      Eigen::VectorXd svals = svd.singularValues();
      // 设置一个奇异值阈值，小于此值的奇异值将被置零（避免数值溢出）
      double threshold = 1e-6 * svals(0); // 例如，最大奇异值的1e-6倍
      // 构造奇异值矩阵的逆
      Eigen::MatrixXd Sinv = MatrixZZ::Zero();
      for (int i = 0; i < svals.size(); ++i) {
          if (svals(i) > threshold) {
              Sinv(i, i) = 1.0 / svals(i);
          } else {
              // 奇异值太小，置零忽略它（或者可以设置一个很小的值）
              Sinv(i, i) = 0.0;
              RCLCPP_WARN(logger, "Small singular value detected and truncated: %e", svals(i));
          }
      }
      // 计算 S 的伪逆
      MatrixZZ robust_S_inverse = svd.matrixV() * Sinv * svd.matrixU().transpose();
      // 然后计算卡尔曼增益
      K = P_pri * H.transpose() * robust_S_inverse;



      // ++++++++++++++++ 新增打印卡尔曼增益 K ++++++++++++++++
      ss_k << K;
      RCLCPP_INFO(logger, "K (Kalman Gain):\n%s", ss_k.str().c_str());
      // +++++++++++++++++++++++++++++++++++++++++++++++++++++

      x_post = x_post + K * (z - z_pri);
      //P_post = (MatrixXX::Identity() - K * H) * P_pri;
      MatrixXX I_KH = MatrixXX::Identity() - K * H;
      P_post = I_KH * P_pri * I_KH.transpose() + K * R * K.transpose();
      // ++++++++++++++++ 新增打印后验协方差 P_post ++++++++++++++++
      ss_p_post << P_post;
      RCLCPP_INFO(logger, "P_post (Updated Covariance):\n%s", ss_p_post.str().c_str());

      ss_x_post << x_post;
      RCLCPP_INFO(logger, "x_post :\n%s", ss_x_post.str().c_str());

      RCLCPP_INFO(logger, "--------------------------------------\n");
      // +++++++++++++++++++++++++++++++++++++++++++++++++++++++
      
      // // 计算卡尔曼增益
      // K = P_pri * H.transpose() * (H * P_pri * H.transpose() + R).inverse();
      // x_post = x_post + K * (z - z_pri);
      // P_post = (MatrixXX::Identity() - K * H) * P_pri;

      P_pri = 0.5 * (P_pri + P_pri.transpose()).eval();
      P_post = 0.5 * (P_post + P_post.transpose()).eval();
    
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
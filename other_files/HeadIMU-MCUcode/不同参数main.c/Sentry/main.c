/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "crc.h"
#include "tim.h"
#include "usb_device.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "usbd_cdc_if.h"
#include "mpu6050_soft.h"
#include <math.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
    float w, x, y, z;
} Quaternion;

typedef struct {
    float q0, q1, q2, q3;  // 四元数
    float kp;              // 比例增益
} IMU_Mahony;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define G 9.81f
#define TO_DEGREE_RATIO 42000.0f
#define TO_RAD_RATIO (TO_DEGREE_RATIO * 180.0f / M_PI)
#define TO_MPERSQ_RATIO (2048.0f / G)
#define SAMPLE_RATE 1000.0f  // 1kHz采样率

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
static IMU_Mahony imu;
static float euler_yaw, euler_pitch, euler_roll;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
static void IMU_Init(IMU_Mahony* imu, float kp, float ax, float ay, float az);
static void IMU_Update(IMU_Mahony* imu, float ax, float ay, float az, float gx, float gy, float gz);
static void QuaternionToEuler(float q0, float q1, float q2, float q3, float* yaw, float* pitch, float* roll);
static void EulerToQuaternion(float yaw, float pitch, float roll, float* q0, float* q1, float* q2, float* q3);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#define USBUartTxDataBufferLen 38

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {
	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_CRC_Init();
	MX_USB_DEVICE_Init();
	MX_TIM3_Init();
	/* USER CODE BEGIN 2 */
	uint8_t USB_uart_tx_data_buffer[USBUartTxDataBufferLen];
	uint32_t USB_uart_tx_data_uint32_block_num = USBUartTxDataBufferLen / 4;
	if (USB_uart_tx_data_uint32_block_num * 4 < USBUartTxDataBufferLen) {
		USB_uart_tx_data_uint32_block_num += 1;
	}

	MPU_Init();
	short received_avx;
	short received_avy;
	short received_avz;
	MPU_Write_Byte(MPU_GYRO_CFG_REG,3<<3); // 设置角速度量程为+-2000°/s
	MPU_Write_Byte(MPU_ACCEL_CFG_REG,3<<3); // 设置加速度量程为+-16g
	MPU_Write_Byte(MPU_SAMPLE_RATE_REG,0); // 设置输出频率=1KHz
	MPU_Write_Byte(MPU_CFG_REG,0); // 设置数字低通滤波器为最小值
	short received_ax;
	short received_ay;
	short received_az;

	for (uint16_t warm_up = 0; warm_up < 1000; warm_up++) {
		MPU_Get_Gyroscope(&received_avx, &received_avy, &received_avz);
		MPU_Get_Accelerometer(&received_ax, &received_ay, &received_az);
	}

	int32_t avx_zero_drift_multiply_8 = -100; // roll 增大逆时针飘
	int32_t avy_zero_drift_multiply_8 = 100; // pitch 增大向上飘
	int32_t avz_zero_drift_multiply_8 = 30; // yaw 增大往右飘

	int32_t rectified_avx_multiply_8;
	int32_t rectified_avy_multiply_8;
	int32_t rectified_avz_multiply_8;


    // 初始化IMU姿态解算器
    float ax = received_ay / TO_MPERSQ_RATIO;
    float ay = -received_ax / TO_MPERSQ_RATIO;
    float az = -received_az / TO_MPERSQ_RATIO;
    IMU_Init(&imu, 0.001f, ax, ay, az);  // kp = 0.001

	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {
		MPU_Get_Gyroscope(&received_avx, &received_avy, &received_avz);
		MPU_Get_Accelerometer(&received_ax, &received_ay, &received_az);

		rectified_avx_multiply_8 = (int32_t)received_avx * 8 - avx_zero_drift_multiply_8;
		rectified_avy_multiply_8 = (int32_t)received_avy * 8 - avy_zero_drift_multiply_8;
		rectified_avz_multiply_8 = (int32_t)received_avz * 8 - avz_zero_drift_multiply_8;

        // 与C++代码完全一致的转换
        // C++: dtheta = np.array([rectified_avz_multiply_8/to_rad_ratio, -rectified_avy_multiply_8/to_rad_ratio, rectified_avz_multiply_8/to_rad_ratio])
        // 修正：第三个元素应该是rectified_avx_multiply_8
        float gx = -rectified_avy_multiply_8 / TO_RAD_RATIO;  // 对应C++代码的dtheta[1]
        float gy = rectified_avx_multiply_8 / TO_RAD_RATIO;   // 对应C++代码的dtheta[2]
        float gz = rectified_avz_multiply_8 / TO_RAD_RATIO;   // 对应C++代码的dtheta[0]

        // C++: acc = np.array([received_ay/to_mpersq_ratio, -received_ax/to_mpersq_ratio, -received_az/to_mpersq_ratio])
        ax = received_ay / TO_MPERSQ_RATIO;
        ay = -received_ax / TO_MPERSQ_RATIO;
        az = -received_az / TO_MPERSQ_RATIO;

        // 更新姿态
        IMU_Update(&imu, ax, ay, az, gx, gy, gz);

        // 获取欧拉角
        QuaternionToEuler(imu.q0, imu.q1, imu.q2, imu.q3,
                         &euler_yaw, &euler_pitch, &euler_roll);

        // 测试反函数
        // float q0, q1, q2, q3;
        // EulerToQuaternion(euler_yaw, euler_pitch, euler_roll, &q0, &q1, &q2, &q3);
        // QuaternionToEuler(q0, q1, q2, q3,
        //                  &euler_yaw, &euler_pitch, &euler_roll);

		memset(USB_uart_tx_data_buffer, 0, USBUartTxDataBufferLen);

		USB_uart_tx_data_buffer[0] = 0xA7;
		USB_uart_tx_data_buffer[1] = 0xB6;
		USB_uart_tx_data_buffer[2] = 0xC5;
		USB_uart_tx_data_buffer[3]  = USBUartTxDataBufferLen - (4 + 4);

		memcpy(USB_uart_tx_data_buffer + 4, &rectified_avx_multiply_8, sizeof(rectified_avx_multiply_8));
		memcpy(USB_uart_tx_data_buffer + 8, &rectified_avy_multiply_8, sizeof(rectified_avy_multiply_8));
		memcpy(USB_uart_tx_data_buffer + 12, &rectified_avz_multiply_8, sizeof(rectified_avz_multiply_8));
		memcpy(USB_uart_tx_data_buffer + 16, &received_ax, sizeof(received_ax));
		memcpy(USB_uart_tx_data_buffer + 18, &received_ay, sizeof(received_ay));
		memcpy(USB_uart_tx_data_buffer + 20, &received_az, sizeof(received_az));
        // 添加欧拉角
        memcpy(USB_uart_tx_data_buffer + 22, &euler_yaw, sizeof(euler_yaw));
        memcpy(USB_uart_tx_data_buffer + 26, &euler_pitch, sizeof(euler_pitch));
        memcpy(USB_uart_tx_data_buffer + 30, &euler_roll, sizeof(euler_roll));

		uint32_t crc32_result = HAL_CRC_Calculate(&hcrc,
				(uint32_t*) USB_uart_tx_data_buffer,
				USB_uart_tx_data_uint32_block_num - 1);
		memcpy(USB_uart_tx_data_buffer + USBUartTxDataBufferLen - 4,
				&crc32_result, 4);
		CDC_Transmit_FS(USB_uart_tx_data_buffer, USBUartTxDataBufferLen);
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
	}
	/* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
	RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
	RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };
	RCC_PeriphCLKInitTypeDef PeriphClkInit = { 0 };

	/** Initializes the RCC Oscillators according to the specified parameters
	 * in the RCC_OscInitTypeDef structure.
	 */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
	RCC_OscInitStruct.HSIState = RCC_HSI_ON;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
	RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL6;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks
	 */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
			| RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK) {
		Error_Handler();
	}
	PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USB;
	PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_PLL;
	if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK) {
		Error_Handler();
	}
}

/* USER CODE BEGIN 4 */

static void IMU_Init(IMU_Mahony* imu, float kp, float ax, float ay, float az) {

	float norm_acc = sqrt(ax * ax + ay * ay + az * az);
    float pitch = asin(-ay / norm_acc);
    float roll = atan2(ax, -az);
    float yaw = 0.0;

    EulerToQuaternion(yaw, pitch, roll, &(imu->q0), &(imu->q1), &(imu->q2), &(imu->q3));

    // imu->q0 = 1.0;
    // imu->q1 = 0.0;
    // imu->q2 = 0.0;
    // imu->q3 = 0.0;
    imu->kp = kp;
}

static void IMU_Update(IMU_Mahony* imu, float ax, float ay, float az, float gx, float gy, float gz) {
    // ---------- 1. 归一化加速度 ----------
    float norm = sqrtf(ax*ax + ay*ay + az*az);
    if (norm > 0.0f) {
        ax /= norm;
        ay /= norm;
        az /= norm;
    }

    float q0 = imu->q0;
    float q1 = imu->q1;
    float q2 = imu->q2;
    float q3 = imu->q3;

    // ---------- 2. 计算重力估计 g_est = R.T * [0, 0, 1] ----------
    // R.T 的第三行即为机体坐标系下Z轴在世界坐标系中的表示的逆(或直接对应重力在机体系的投影)
    // R[2,0] = 2*(x*z - y*w)
    // R[2,1] = 2*(y*z + x*w)
    // R[2,2] = 1 - 2*(x*x + y*y)
    float gx_est = 2.0f * (q1 * q3 - q0 * q2);
    float gy_est = 2.0f * (q2 * q3 + q0 * q1);
    float gz_est = 1.0f - 2.0f * (q1 * q1 + q2 * q2);

    // ---------- 3. Mahony 姿态修正 ----------
    // error = cross(g_est, acc_measured)
    float ex = gy_est * az - gz_est * ay;
    float ey = gz_est * ax - gx_est * az;
    float ez = gx_est * ay - gy_est * ax;// 0.0f; // 不允许修正 yaw (error[2] = 0)

    // 融合陀螺仪数据与修正项 (假设 gx, gy, gz 为积分过的 dtheta 或包含 dt 因子的角速度)
    // 若 gx, gy, gz 为原始角速度(rad/s)，这里需要在外部乘以 dt 或在此处乘以 dt
    // 这里按照 Python 代码逻辑直接使用，假设调用时已经处理了时间步长
    float omega_x = gx + imu->kp * ex;
    float omega_y = gy + imu->kp * ey;
    float omega_z = gz + imu->kp * ez;

    // ---------- 4. 欧拉法姿态更新 ----------
    // q_dot = 0.5 * q * omega
    float q0_dot = 0.5f * (-q1 * omega_x - q2 * omega_y - q3 * omega_z);
    float q1_dot = 0.5f * ( q0 * omega_x + q2 * omega_z - q3 * omega_y);
    float q2_dot = 0.5f * ( q0 * omega_y - q1 * omega_z + q3 * omega_x);
    float q3_dot = 0.5f * ( q0 * omega_z + q1 * omega_y - q2 * omega_x);

    imu->q0 += q0_dot;
    imu->q1 += q1_dot;
    imu->q2 += q2_dot;
    imu->q3 += q3_dot;

    // ---------- 5. 归一化四元数 ----------
    norm = sqrtf(imu->q0*imu->q0 + imu->q1*imu->q1 + imu->q2*imu->q2 + imu->q3*imu->q3);
    if (norm > 0.0f) {
        imu->q0 /= norm;
        imu->q1 /= norm;
        imu->q2 /= norm;
        imu->q3 /= norm;
    }
}

static void QuaternionToEuler(float q0, float q1, float q2, float q3, float* yaw, float* pitch, float* roll) {
    // 对应 Python 中的 quat_to_euler_local_ZXY
    // Intrinsic Z-X-Y (Yaw-Pitch-Roll)

    // 旋转矩阵元素 R (world to body or body to world consistency check)
    // Python Code:
    // pitch = np.arcsin(R[2,1])             -> 2*(q2*q3 + q0*q1)
    // yaw   = np.arctan2(-R[0,1], R[1,1])   -> atan2(-2*(q1*q2 - q0*q3), 1 - 2*(q1*q1 + q3*q3))
    // roll  = np.arctan2(-R[2,0], R[2,2])   -> atan2(-2*(q1*q3 - q0*q2), 1 - 2*(q1*q1 + q2*q2))

    *pitch = asinf(2.0f * (q2 * q3 + q0 * q1));
    *yaw   = atan2f(-2.0f * (q1 * q2 - q0 * q3), 1.0f - 2.0f * (q1 * q1 + q3 * q3));
    *roll  = atan2f(-2.0f * (q1 * q3 - q0 * q2), 1.0f - 2.0f * (q1 * q1 + q2 * q2));
}
static void EulerToQuaternion(float yaw, float pitch, float roll, float* q0, float* q1, float* q2, float* q3) {
	float half_yaw = yaw / 2.0;
	float half_pitch = pitch / 2.0;
	float half_roll = roll / 2.0;

	float cy = cos(half_yaw);
	float sy = sin(half_yaw);
	float cp = cos(half_pitch);
	float sp = sin(half_pitch);
	float cr = cos(half_roll);
	float sr = sin(half_roll);

	*q0 = cy * cp * cr - sy * sp * sr;
	*q1 = cy * sp * cr - sy * cp * sr;
	*q2 = cy * cp * sr + sy * sp * cr;
	*q3 = cy * sp * sr + sy * cp * cr;
}


/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
	/* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

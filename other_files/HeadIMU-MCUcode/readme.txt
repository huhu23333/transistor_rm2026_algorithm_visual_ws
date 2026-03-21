使用 STM32CubeIDE 1.10.1
这个工程一开始只是用来测试stm32的虚拟串口用的，后来直接改成IMU的驱动了，名字懒得改了
使用stm32f103c6t6(c8t6应该也能用) + mpu6050
mpu6050 <-> stm32
VCC <-> 3.3
GND/AD0 <-> GND
SCL <-> PB6
SDA <-> PB7


# 北航Transistor战队RM2026算法视觉组代码

## 0、安装

* 需提前安装好ROS2 Humble和MVS（海康机器人工业相机驱动），可参考下列链接安装：

```
https://fishros.com/d2lros2/#/humble/chapt1/get_started/3.%E5%8A%A8%E6%89%8B%E5%AE%89%E8%A3%85ROS2
https://www.hikrobotics.com/cn/machinevision/service/download/
```

* 最外层目录（即该README.md所在目录）需保持命名为"transistor_rm2026_algorithm_visual_ws"，配置文件读取依赖此路径名。
* 下列命令的执行路径均为上述最外层文件夹。
* 运行下命令自动安装必要依赖（该命令会在~目录下建立"makeInstall"文件夹，所有需要编译安装的依赖保存于此）：

```
sudo bash ./pkgInstall/pkgInstall.bash
```

* 如有Intel的GPU（包括核显），可执行下命令安装驱动：

```
sudo bash ./pkgInstall/intelGpuDriverInstall.bash
```

* 安装失败可尝试按照`./pkgInstall/pkgInstall.txt`中的命令和操作手动安装。

## 1、配置

* 主要配置文件为下列两个文件：

```
src/shared_files/config.yaml
src/auto_aim/include/macro/AutoAimMacro.h
```

* 其中config.yaml更改后无需重新编译，AutoAimMacro.h更改后需重新编译才能生效。
* 无摄像头时可在`AutoAimMacro.h`中定义`USE_VIDEO`或`USE_IMAGES`后重新编译，切换至使用视频或图片文件夹输入，使用的视频或图片文件夹路径在`config.yaml`中。

## 2、编译

* 运行下命令编译：

```
colcon build
```

## 3、运行

* 运行下命令导入环境（每次打开终端后该命令仅需运行一次，多次运行无需重复执行）：

```
source ./install/setup.bash
```

* 运行下命令运行：

```
ros2 launch auto_aim auto_aim_launch.py
```

* 运行后可能没有画面，需定义`AutoAimMacro.h`中的`SHOW_WINDOWS`后重新编译。

## 4、自启动

* 运行下命令安装自启动服务：

```
sudo ./auto_launch/serviceInstall.py
```

* 该命令会产生自启动服务文件`/etc/systemd/system/auto_aim_auto_launch.service`。
* 使用自启动时需取消定义`AutoAimMacro.h`中的`SHOW_WINDOWS`后重新编译，否则无法自启动。
* 自启动使用`auto_launch`中的看门狗启动主程序，而直接运行不使用看门狗。可手动运行`./auto_launch/watchdog.py`来使用看门狗启动主程序。
* 使用下列命令管理自瞄程序自启动：

```
sudo systemctl enable auto_aim_auto_launch.service # 启用自启动
sudo systemctl disable auto_aim_auto_launch.service # 取消自启动
sudo systemctl start auto_aim_auto_launch.service # 单次运行自启动脚本
sudo systemctl stop auto_aim_auto_launch.service # 关闭当前运行中的自启动程序
sudo systemctl status auto_aim_auto_launch.service # 查看自启动服务状态
```

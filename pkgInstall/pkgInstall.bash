#!/bin/bash
sudo usermod -a -G dialout $USER
sudo chmod a+rw /dev/ttyACM0
cd /etc/udev/rules.d
sudo touch 99-usb-serial.rules
KERNEL=="ttyACM0", MODE="0666"
sudo udevadm control --reload-rules
sudo udevadm trigger

sudo apt update
sudo apt install -y python3-pip
sudo apt install -y libeigen3-dev
sudo apt install -y libceres-dev
sudo apt install -y libfmt-dev

pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy==2.2.6 sysv-ipc==1.1.0 opencv-python==4.12.0.88 pyyaml==5.4.1
pip3 install ultralytics==8.4.5 openvino==2024.6.0
pip3 install onnxruntime==1.23.2 onnxscript==0.5.6 openvino-dev==2024.6.0
pip3 install nncf==2.14.0
pip3 install posix_ipc==1.3.2


# cmake 3.24
cd ~
mkdir makeInstall && cd makeInstall
mkdir cmake && cd cmake
wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.tar.gz
tar -xvf ./cmake-3.24.1-linux-x86_64.tar.gz

# Sophus库 (G2O库依赖)
cd ~/makeInstall
git clone https://github.com/strasdat/Sophus
cd Sophus
sed -i.bak '/option(BUILD_SOPHUS_TESTS/s/ON/OFF/' CMakeLists.txt
mkdir build && cd build
~/makeInstall/cmake/cmake-3.24.1-linux-x86_64/bin/cmake ..
make -j4
sudo make install

# G2O库 (优化装甲板Yaw角度)
cd ~/makeInstall
sudo apt install libspdlog-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5
git clone https://github.com/RainerKuemmerle/g2o
cd g2o
mkdir build && cd build
~/makeInstall/cmake/cmake-3.24.1-linux-x86_64/bin/cmake ..
make -j4
sudo make install

CONF_FILE="/etc/ld.so.conf"
NEW_LINE="/usr/local/lib"
# 检查是否已有该行（忽略前后空白，避免重复添加）
if grep -qxF "$NEW_LINE" "$CONF_FILE"; then
    echo "配置行 '$NEW_LINE' 已经存在于 $CONF_FILE 中，无需添加。"
else
    # 备份原配置文件
    sudo cp "$CONF_FILE" "${CONF_FILE}.bak"
    echo "已备份原配置文件至 ${CONF_FILE}.bak"

    # 添加新行
    echo "$NEW_LINE" | sudo tee -a "$CONF_FILE" > /dev/null
    echo "已成功添加 '$NEW_LINE' 到 $CONF_FILE"

fi

# 更新动态链接库缓存
echo "正在运行 ldconfig 更新缓存..."
sudo ldconfig
echo "ldconfig 执行完成。"

# 视频日志查看查看工具
sudo apt install vlc

cd ~/makeInstall
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB  
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB  
echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list  
sudo apt update 
apt-cache search openvino
sudo apt install openvino-2024.0.0

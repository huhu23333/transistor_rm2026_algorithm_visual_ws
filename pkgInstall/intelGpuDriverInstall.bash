#!/bin/bash
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt update
sudo apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo
sudo apt-get install -y libze-dev intel-ocloc

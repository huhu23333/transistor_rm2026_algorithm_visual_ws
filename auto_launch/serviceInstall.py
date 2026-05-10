#!/usr/bin/env python3

#/etc/systemd/system/auto_aim_auto_launch.service
#sudo systemctl enable/disable/start/stop/status auto_aim_auto_launch.service

import os
import sys
import subprocess

original_service_file_content = """
[Unit]
Description=Auto Aim Auto Launch Service
After=network.target network-online.target

[Service]
Type=simple
User=rm1
ExecStart=/bin/bash -c "/home/rm1/rm2026/transistor_rm2026_algorithm_visual_ws/auto_launch/watchdog.py"
WorkingDirectory=/home/rm1/rm2026/transistor_rm2026_algorithm_visual_ws

[Install]
WantedBy=multi-user.target
"""

after_commands = """
sudo chmod +x ./auto_aim_launch.py
sudo chmod +x ./watchdog.py
sudo systemctl daemon-reload
sudo systemctl enable auto_aim_auto_launch.service
"""

may_user_list = os.listdir("/home")
if len(may_user_list) == 0:
    sys.exit()
elif len(may_user_list) == 1:
    user_name = may_user_list[0]
else:
    for may_user_name in may_user_list:
        if "rm2026" in os.listdir("/home/"+may_user_name):
            user_name = may_user_name
            break
    else:
        user_name = input("请输入用户名：")

auto_launch_dir = os.path.dirname(__file__)
ws_dir = auto_launch_dir.replace("/auto_launch", "")
changed_service_file_content = original_service_file_content\
    .replace("/home/rm1/rm2026/transistor_rm2026_algorithm_visual_ws", ws_dir)\
    .replace("User=rm1", f"User={user_name}")
temp_file_name = os.path.join(auto_launch_dir, "auto_aim_auto_launch.service")
with open(temp_file_name, mode="w", encoding="utf-8") as temp_file:
    temp_file.write(changed_service_file_content)

commands = f"""cd {auto_launch_dir}
sudo mv ./auto_aim_auto_launch.service /etc/systemd/system/
"""
commands += after_commands

result = subprocess.run(
    ["bash", "-c", commands]
)



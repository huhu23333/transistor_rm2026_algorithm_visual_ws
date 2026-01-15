#!/usr/bin/env python3

# sleep 10
# source /opt/ros/humble/setup.bash
# source /home/rm1/rm2026/transistor_rm2026_algorithm_visual_ws/install/setup.bash
# ros2 launch auto_aim auto_aim_launch.py

import os
import sys
import subprocess

original_command = """source /opt/ros/humble/setup.bash
source /home/rm1/rm2026/transistor_rm2026_algorithm_visual_ws/install/setup.bash
ros2 launch auto_aim auto_aim_launch.py"""

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
        sys.exit()

changed_command = original_command.replace("rm1", user_name)

result = subprocess.run(
    ["bash", "-c", changed_command]
)

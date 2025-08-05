# auto_aim_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='auto_aim',
            executable='com_node',  # 改为 com_node
            name='com_node',        # 改为 com_node
            parameters=[{
                'serial_port': '/dev/ttyACM0',
                'baudrate': 115200
            }]
        ),
        Node(
            package='auto_aim',
            executable='armor_detect_node',
            name='armor_detect_node'
        )
    ])
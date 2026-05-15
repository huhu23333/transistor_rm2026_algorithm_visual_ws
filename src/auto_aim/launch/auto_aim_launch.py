# auto_aim_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import openvino as ov

def generate_launch_description():
    # openvino_core = ov.Core()
    # if "GPU" in openvino_core.available_devices:
    #     shm_yolo_pose_node = ExecuteProcess(
    #         cmd=['taskset', '-c', "0,1", 'ros2', 'run', 'shm_python_processor_pkg', 'shm_yolo_pose_node'],
    #         output='screen'
    #     )
    # else:
    #     shm_yolo_pose_node = Node(
    #         package='shm_python_processor_pkg',
    #         executable='shm_yolo_pose_node',
    #         name='shm_yolo_pose_node',
    #         #arguments = ['--ros-args', '--log-level', 'DEBUG']
    #     )

    return LaunchDescription([
        #Node(
        #    package='auto_aim',
        #    executable='com_node',  # 改为 com_node
        #    name='com_node',        # 改为 com_node
        #    parameters=[{
        #        'serial_port': '/dev/ttyACM0',
        #        'baudrate': 115200
        #    }]
        #),
        Node(
            package='auto_aim',
            executable='armor_detect_node',
            name='armor_detect_node',
            output='screen',  # <--- 加上这一行
            #arguments = ['--ros-args', '--log-level', 'DEBUG']
        ),
        Node(
            package='shm_python_processor_pkg',
            executable='shm_classifier_node',
            name='shm_classifier_node',
            #arguments = ['--ros-args', '--log-level', 'DEBUG']
        ),
        # shm_yolo_pose_node
    ])
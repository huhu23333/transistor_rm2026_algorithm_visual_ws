#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

class ROS2Watchdog:
    def __init__(self, launch_script, socket_path="/tmp/ros2_watchdog.sock", 
                 feed_timeout=5.0, connect_timeout=10.0):
        """
        Args:
            launch_script: 启动C++程序的shebang脚本路径
            socket_path: Unix域套接字路径
            feed_timeout: 喂狗超时时间(秒)
            connect_timeout: 连接超时时间(秒)
        """
        self.launch_script = Path(launch_script).resolve()
        self.socket_path = socket_path
        self.feed_timeout = feed_timeout
        self.connect_timeout = connect_timeout
        
        self.process = None
        self.last_feed_time = 0
        self.running = False
        self.watchdog_thread = None
        
        # 确保套接字文件不存在
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
    
    def start_program(self):
        """启动C++程序"""
        if self.process and self.process.poll() is None:
            self.force_kill()
        
        print(f"启动程序: {self.launch_script}")
        
        # 设置环境变量传递套接字路径
        env = os.environ.copy()
        env['WATCHDOG_SOCKET_PATH'] = self.socket_path
        
        try:
            self.process = subprocess.Popen(
                [str(self.launch_script)],
                cwd=self.launch_script.parent,
                env=env,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            self.last_feed_time = time.time()
            return True
        except Exception as e:
            print(f"启动失败: {e}")
            return False
    
    def force_kill(self):
        """强制终止程序"""
        if self.process:
            try:
                # 使用进程组终止所有相关进程
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=2)
                print("程序已强制终止")
            except:
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        self.force_kill()
        
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except:
                pass
    
    def run(self):
        """运行看门狗主循环"""
        self.running = True
        
        while self.running:
            # 启动或重启程序
            if not self.process or self.process.poll() is not None:
                if not self.start_program():
                    print("程序启动失败，5秒后重试...")
                    time.sleep(5)
                    continue
            
            # 等待连接
            print("等待程序连接...")
            connect_start = time.time()
            connected = False
            
            while time.time() - connect_start < self.connect_timeout:
                if os.path.exists(self.socket_path):
                    connected = True
                    break
                time.sleep(0.1)
                
                # 检查程序是否已经退出
                if self.process.poll() is not None:
                    print("程序在连接前已退出")
                    break
            
            if not connected:
                print(f"连接超时({self.connect_timeout}秒)")
                self.force_kill()
                continue
            
            print("连接建立，开始监控...")
            
            # 监控循环
            while self.running:
                # 检查进程是否存活
                if self.process.poll() is not None:
                    print("程序异常退出")
                    break
                
                # 检查喂狗超时
                current_time = time.time()
                elapsed = current_time - self.last_feed_time
                
                if elapsed > self.feed_timeout:
                    print(f"喂狗超时({elapsed:.1f}秒 > {self.feed_timeout}秒)")
                    self.force_kill()
                    break
                
                # 更新喂狗时间（如果文件被修改）
                try:
                    mtime = os.path.getmtime(self.socket_path)
                    if mtime > self.last_feed_time:
                        self.last_feed_time = mtime
                except:
                    pass
                
                time.sleep(0.1)
    
    def start(self):
        """启动看门狗"""
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            self.run()
        except KeyboardInterrupt:
            print("\n收到中断信号")
        except Exception as e:
            print(f"看门狗异常: {e}")
        finally:
            self.cleanup()
    
    def signal_handler(self, sig, frame):
        """信号处理函数"""
        print(f"收到信号 {sig}，正在关闭...")
        self.running = False

def main():
    # 配置参数
    LAUNCH_SCRIPT = "./run_ros2_program"  # shebang启动脚本的相对路径
    SOCKET_PATH = "/tmp/ros2_watchdog.sock"
    FEED_TIMEOUT = 5.0    # 5秒喂狗超时
    CONNECT_TIMEOUT = 10.0 # 10秒连接超时
    
    # 创建并运行看门狗
    watchdog = ROS2Watchdog(
        launch_script=LAUNCH_SCRIPT,
        socket_path=SOCKET_PATH,
        feed_timeout=FEED_TIMEOUT,
        connect_timeout=CONNECT_TIMEOUT
    )
    
    watchdog.start()

if __name__ == "__main__":
    main()

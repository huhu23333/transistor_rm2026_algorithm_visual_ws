#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import threading
import socket
import select
from pathlib import Path

class RM2026VisionWatchdog:
    def __init__(self, launch_script, socket_path="/tmp/rm2026_vision_watchdog.sock", 
                 feed_timeout=30.0, connect_timeout=30.0):
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
        self.server_socket = None
        self.client_connection = None  # 初始化client_connection
        
        # 确保套接字文件不存在
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
    
    def create_server_socket(self):
        """创建Unix域套接字服务器"""
        try:
            # 删除已存在的套接字文件
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            
            # 创建服务器套接字
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(1)  # 只允许一个客户端连接
            self.server_socket.settimeout(1.0)  # 设置超时，以便可以检查是否停止
            
            # 设置套接字文件权限，确保其他用户可以连接（如果需要在不同用户下运行）
            os.chmod(self.socket_path, 0o666)
            
            print(f"创建看门狗套接字: {self.socket_path}")
            return True
        except Exception as e:
            print(f"创建服务器套接字失败: {e}")
            return False
    
    def accept_client(self, timeout=2.0):
        """接受客户端连接"""
        if not self.server_socket:
            return False
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 设置非阻塞模式接受连接
                self.server_socket.settimeout(0.1)
                conn, _ = self.server_socket.accept()
                conn.settimeout(0.1)  # 设置接收超时
                self.client_connection = conn
                print("客户端已连接")
                return True
            except socket.timeout:
                # 超时，继续循环
                if not self.running:
                    return False
                continue
            except Exception as e:
                print(f"接受客户端连接失败: {e}")
                return False
        
        return False
    
    def handle_client_messages(self):
        """处理客户端消息"""
        if not self.client_connection:
            return False
        
        try:
            # 设置非阻塞模式读取数据
            ready = select.select([self.client_connection], [], [], 0.1)
            if ready[0]:
                data = self.client_connection.recv(1024)
                if data:
                    # 解码并处理消息
                    message = data.decode('utf-8').strip()
                    # 更新最后喂狗时间
                    self.last_feed_time = time.time()
                    # 可选：打印收到的心跳消息
                    # print(f"收到心跳: {message}")
                    return True
        except socket.timeout:
            # 正常超时，继续
            pass
        except Exception as e:
            print(f"接收客户端消息失败: {e}")
            # 连接可能已断开
            self.client_connection.close()
            self.client_connection = None
            return False
        
        return False
    
    def start_program(self):
        """启动C++程序"""
        if self.process and self.process.poll() is None:
            self.force_kill()
        
        # 重置连接状态
        if self.client_connection:
            try:
                self.client_connection.close()
            except:
                pass
            self.client_connection = None
        
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
        
        # 关闭客户端连接
        if hasattr(self, 'client_connection') and self.client_connection:
            try:
                self.client_connection.close()
            except:
                pass
            self.client_connection = None
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        
        # 关闭套接字连接
        if hasattr(self, 'client_connection') and self.client_connection:
            try:
                self.client_connection.close()
            except:
                pass
            self.client_connection = None
        
        # 关闭服务器套接字
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        
        # 删除套接字文件
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except:
                pass
        
        # 终止程序
        self.force_kill()
    
    def run(self):
        """运行看门狗主循环"""
        self.running = True
        
        # 创建服务器套接字
        if not self.create_server_socket():
            print("无法创建服务器套接字，退出...")
            return
        
        while self.running:
            # 启动或重启程序
            if not self.process or self.process.poll() is not None:
                if not self.start_program():
                    print("程序启动失败，5秒后重试...")
                    time.sleep(5)
                    continue
            
            print("等待程序连接...")
            connect_start = time.time()
            connected = False
            
            # 等待客户端连接
            while time.time() - connect_start < self.connect_timeout:
                if self.accept_client(timeout=0.5):
                    connected = True
                    break
                
                # 检查程序是否已经退出
                if self.process.poll() is not None:
                    print("程序在连接前已退出")
                    break
            
            if not connected:
                print(f"连接超时({self.connect_timeout}秒)")
                self.force_kill()
                continue
            
            print("连接建立，开始监控...")
            self.last_feed_time = time.time()  # 连接建立时重置喂狗时间
            
            # 监控循环
            while self.running:
                # 检查进程是否存活
                if self.process.poll() is not None:
                    print("程序异常退出")
                    break
                
                # 处理客户端消息（更新喂狗时间）
                self.handle_client_messages()
                
                # 检查喂狗超时
                current_time = time.time()
                elapsed = current_time - self.last_feed_time
                
                if elapsed > self.feed_timeout:
                    print(f"喂狗超时({elapsed:.1f}秒 > {self.feed_timeout}秒)")
                    self.force_kill()
                    break
                
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
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def signal_handler(self, sig, frame):
        """信号处理函数"""
        print(f"收到信号 {sig}，正在关闭...")
        self.cleanup()

def main():
    # 配置参数
    LAUNCH_SCRIPT = "auto_aim_launch.py"  # shebang启动脚本的名字（要求在同一目录下）
    SOCKET_PATH = "/tmp/rm2026_vision_watchdog.sock"
    CONNECT_TIMEOUT = 15.0 # 15秒连接超时
    FEED_TIMEOUT = 15.0    # 15秒喂狗超时

    LAUNCH_SCRIPT = Path(os.path.dirname(__file__)) / LAUNCH_SCRIPT
    
    # 创建并运行看门狗
    watchdog = RM2026VisionWatchdog(
        launch_script=LAUNCH_SCRIPT,
        socket_path=SOCKET_PATH,
        feed_timeout=FEED_TIMEOUT,
        connect_timeout=CONNECT_TIMEOUT
    )
    
    watchdog.start()

if __name__ == "__main__":
    main()

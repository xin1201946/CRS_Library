import datetime
import platform
import socket
import sys
import winreg
import psutil
import GPUtil
import time
import requests
import json
from rich.console import Console
import wifi
ip = ""

def get_system_info():
    """
    获取系统信息，返回字典格式
    Get system information and return it in dictionary format.
    """
    system_info = {
        'info': 'Normal',
        'python': {
            # Python 版本信息
            # Python version information
            'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            # Python 发布级别
            # Python release level
            'released': f"{sys.version_info.releaselevel}",
            # Python 修复编号
            # Python repair number
            'RepairNumber': f"{sys.version_info.serial}",
        },
        'os': {
            # 操作系统平台
            # Operating system platform
            'platform': platform.system(),
            # 操作系统版本
            # Operating system release
            'release': platform.release(),
            # 操作系统详细版本信息
            # Operating system detailed version information
            'version': platform.version()
        },
        # 主机名
        # Hostname
        'hostname': socket.gethostname()
    }

    try:
        # 获取外部 IP 地址
        # Get the external IP address
        system_info['external_ip'] = ip if get_ip() == "" or get_ip() is None else get_ip()
    except Exception as e:
        # 若获取失败，记录错误信息
        # If the acquisition fails, record the error information
        system_info['external_ip'] = {'error': str(e)}

    # 当前服务器时间 (不需要日期, 格式为 00:00 即可)
    # Current server time (no date required, format: 00:00)
    now = datetime.datetime.now()
    system_info['current_time'] = now.strftime("%H:%M:%S")

    # 运行时长
    # Uptime
    system_info['uptime'] = get_system_uptime()

    # 内存信息
    # Memory information
    mem = psutil.virtual_memory()
    system_info['memory'] = {
        # 总内存（MB）
        # Total memory (MB)
        'total': mem.total / (1024 * 1024),
        # 已使用内存（MB）
        # Used memory (MB)
        'used': mem.used / (1024 * 1024),
        # 可用内存（MB）
        # Available memory (MB)
        'available': mem.available / (1024 * 1024),
        # 内存使用率
        # Memory usage percentage
        'percent': mem.percent
    }
    swap = psutil.swap_memory()
    system_info['swap'] = {
        # 总交换空间（MB）
        # Total swap space (MB)
        'total': swap.total / (1024 * 1024),
        # 已使用交换空间（MB）
        # Used swap space (MB)
        'used': swap.used / (1024 * 1024),
        # 空闲交换空间（MB）
        # Free swap space (MB)
        'free': swap.free / (1024 * 1024),
        # 交换空间使用率
        # Swap space usage percentage
        'percent': swap.percent
    }

    # 处理器信息
    # Processor information
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    # QueryValueEx 获取指定注册表中指定字段的内容
    # QueryValueEx gets the content of the specified field in the specified registry
    cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")  # 获取cpu名称
    # Get the CPU name
    key.Close()
    # 物理 CPU 核心数
    # Number of physical CPU cores
    cpu_count = psutil.cpu_count(logical=False)
    # CPU 使用率
    # CPU usage percentage
    cpu_percent = psutil.cpu_percent(interval=1)
    # CPU 时间统计
    # CPU time statistics
    cpu_times = psutil.cpu_times()
    system_info['cpu'] = {
        # CPU 名称
        # CPU name
        'name': cpu_name[0],
        # CPU 核心数
        # Number of CPU cores
        'count': cpu_count,
        # CPU 使用率
        # CPU usage percentage
        'percent': cpu_percent,
        'times': {
            # 用户态 CPU 时间
            # User mode CPU time
            'user': cpu_times.user,
            # 系统态 CPU 时间
            # System mode CPU time
            'system': cpu_times.system,
            # 空闲 CPU 时间
            # Idle CPU time
            'idle': cpu_times.idle
        }
    }

    # 显卡信息
    # Graphics card information
    try:
        gpus = GPUtil.getGPUs()
        system_info['gpus'] = []
        for gpu in gpus:
            system_info['gpus'].append({
                # 显卡 ID
                # Graphics card ID
                'id': gpu.id,
                # 显卡名称
                # Graphics card name
                'name': gpu.name,
                # 显卡驱动版本
                # Graphics card driver version
                'driver': gpu.driver,
                # 显卡总内存（GB）
                # Total graphics card memory (GB)
                'memory_total': gpu.memoryTotal / 1024,
                # 已使用显卡内存（GB）
                # Used graphics card memory (GB)
                'memory_used': gpu.memoryUsed / 1024,
                # 空闲显卡内存（GB）
                # Free graphics card memory (GB)
                'memory_free': gpu.memoryFree / 1024,
                # 显卡内存使用率
                # Graphics card memory usage percentage
                'memory_percent': gpu.memoryUtil * 100,
                # 显卡温度
                # Graphics card temperature
                'temperature': gpu.temperature,
                # 显卡负载
                # Graphics card load
                'load': gpu.load * 100
            })
    except Exception as e:
        # 若获取失败，记录错误信息
        # If the acquisition fails, record the error information
        system_info['gpus'] = {'error': str(e)}

    # 网络信息
    # Network information
    system_info['network'] = get_network_info()

    return system_info


def get_ip():
    """
    获取外部 IP 地址
    Get the external IP address.
    """
    response = requests.get('https://api.vore.top/api/IPdata?ip=', timeout=2)
    try:
        response = json.loads(response.text)
    except Exception:
        response = {'ipinfo': {"text": "Timeout"}}
    finally:
        return response["ipinfo"]["text"]


def get_system_uptime():
    """
    计算系统的运行时间
    Calculate the system uptime.
    """
    # 计算系统的运行时间（以秒为单位）
    # Calculate the system uptime in seconds
    uptime = time.time() - psutil.boot_time()
    # 创建一个 timedelta 对象来表示运行时间
    # Create a timedelta object to represent the uptime
    delta = datetime.timedelta(seconds=uptime)
    # 格式化输出，精确到秒
    # Format the output to the nearest second
    return f"{delta.days} days, {delta.seconds // 3600:02d}:{delta.seconds % 3600 // 60:02d}:{delta.seconds % 60:02d}"

def send_sysInfo(socketios, get_clients):
    """
    循环发送系统信息给客户端
    Continuously send system information to clients.

    :param socketios: SocketIO 对象
    :param socketios: SocketIO object
    :param get_clients: 获取客户端列表的函数
    :param get_clients: Function to get the list of clients
    """
    while True:
        try:
            # 获取系统信息
            # Get system information
            clients = get_clients()
            system_info = get_system_info()
            if not system_info:
                continue

            # 将用户数添加到系统信息中
            # Add the number of users to the system information
            system_info['Usercount'] = len(clients)

            # 转换为 JSON 字符串
            # Convert to JSON string
            system_info_json = json.dumps(system_info)

            # 发送信息给每个客户端
            # Send information to each client
            for uuid, sid in clients.items():
                try:
                    socketios.emit('sysinfo_update', system_info_json, to=sid)
                except Exception as e:
                    print(f"Error sending to {uuid}: {e}")

            time.sleep(1)
        except Exception as e:
            time.sleep(1)  # 等待一段时间后再次尝试
            # Wait for a while and try again


def get_network_info():
    """
    获取网络信息
    Get network information.
    """
    network_info = []
    # 获取系统中所有网络接口
    # Get all network interfaces in the system
    for iface, stats in psutil.net_if_stats().items():
        if iface.lower().startswith(('lo', '蓝牙', 'blue', 'vm', 'veth', 'docker', 'tun', 'br')):  # 可以根据实际情况扩展
            # 可以根据实际情况扩展
            # Can be extended according to actual situation
            continue

        interface_info = {'name': iface}

        # 获取网卡的IO计数
        # Get the network card's IO counters
        net_io_counters = psutil.net_io_counters(pernic=True).get(iface)
        if net_io_counters:
            interface_info.update({
                # 接收字节数
                # Received bytes
                'bytes_recv': net_io_counters.bytes_recv,
                # 发送字节数
                # Sent bytes
                'bytes_sent': net_io_counters.bytes_sent,
                # 接收数据包数
                # Received packets
                'packets_recv': net_io_counters.packets_recv,
                # 发送数据包数
                # Sent packets
                'packets_sent': net_io_counters.packets_sent,
            })

        # 获取网卡的IP地址信息
        # Get the network card's IP address information
        addrs = psutil.net_if_addrs().get(iface, [])
        ip4, ip6 = None, None
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ip4 = addr.address
            elif addr.family == socket.AF_INET6:
                ip6 = addr.address

        interface_info['ipv4'] = ip4
        interface_info['ipv6'] = ip6

        # 获取网卡的连接状态和类型
        # Get the network card's connection status and type
        interface_info['is_up'] = stats.isup
        # 只检测活动网卡
        # Only detect active network cards
        if stats.isup:
            interface_info['speed'] = get_network_usage(iface, stats.speed)
        else:
            # 非活动网卡返回最大速率
            # Return the maximum speed for inactive network cards
            interface_info['speed'] = stats.speed

        # 如果是WiFi，尝试获取SSID和信号强度
        # If it is a WiFi network, try to get the SSID and signal strength
        if iface.startswith('wlan'):
            wifi_info = get_wifi_info(iface)
            interface_info.update(wifi_info)

        network_info.append(interface_info)

    return network_info


def get_wifi_info(iface):
    """
    获取 WiFi 信息
    Get WiFi information.

    :param iface: 网络接口名称
    :param iface: Network interface name
    """
    wifi_info = {}
    try:
        wifi_details = wifi.Cell.all(iface)
        if wifi_details:
            # 获取连接的WiFi网络的SSID和信号强度
            # Get the SSID and signal strength of the connected WiFi network
            wifi_info['ssid'] = wifi_details[0].ssid
            wifi_info['signal_strength'] = wifi_details[0].signal
    except Exception as e:
        wifi_info['ssid'] = None
        wifi_info['signal_strength'] = None

    return wifi_info


def get_network_usage(iface, MAX_BANDWIDTH):
    """
    计算网络接口的使用情况
    Calculate the network interface usage.

    :param iface: 网络接口名称
    :param iface: Network interface name
    :param MAX_BANDWIDTH: 最大带宽
    :param MAX_BANDWIDTH: Maximum bandwidth
    """
    net_io_before = psutil.net_io_counters(pernic=True).get(iface)

    # 如果没有找到网卡的IO信息，返回默认值
    # If no network card IO information is found, return default values
    if not net_io_before:
        return {
            # 下载速度（kB/s）
            # Download speed (kB/s)
            "download_speedKBps": "0.00",
            # 上传速度（kB/s）
            # Upload speed (kB/s)
            "upload_speedKBps": "0.00",
            # 总速率（kB/s）
            # Total speed (kB/s)
            "total_speedKBps": "0.00",
            # 占用率百分比
            # Utilization percentage
            "utilization": "0.0",
        }

    # 等待一定的时间来计算流量差
    # Wait for a certain period of time to calculate the traffic difference
    time.sleep(1)

    # 获取时间间隔后的网络IO信息
    # Get the network IO information after the time interval
    net_io_after = psutil.net_io_counters(pernic=True).get(iface)

    # 计算每秒的接收和发送字节数变化
    # Calculate the change in received and sent bytes per second
    bytes_recv = net_io_after.bytes_recv - net_io_before.bytes_recv
    bytes_sent = net_io_after.bytes_sent - net_io_before.bytes_sent

    # 转换为kB/s
    # Convert to kB/s
    recv_speed_kbps = bytes_recv / 1024  # 下载速度（kB/s）
    send_speed_kbps = bytes_sent / 1024  # 上传速度（kB/s）

    # 计算总速率（单位：kB/s）
    # Calculate the total speed (in kB/s)
    total_speed_kbps = recv_speed_kbps + send_speed_kbps

    # 计算网卡的占用率
    # Calculate the network card utilization
    utilization = (total_speed_kbps / (MAX_BANDWIDTH * 1024)) * 100  # 占用率百分比

    return {
        # 下载速度，单位 kB/s
        # Download speed, in kB/s
        "download_speedKBps": f"{recv_speed_kbps:.2f}",
        # 上传速度，单位 kB/s
        # Upload speed, in kB/s
        "upload_speedKBps": f"{send_speed_kbps:.2f}",
        # 总速率，单位 kB/s
        # Total speed, in kB/s
        "total_speedKBps": f"{total_speed_kbps:.2f}",
        # 占用率百分比
        # Utilization percentage
        "utilization": f"{utilization:.1f}",
    }

if __name__ == "__main__":
    console = Console()
    # 打印系统信息
    # Print system information
    console.print(get_system_info())
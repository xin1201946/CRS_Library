import datetime
import platform
import socket
import sys
import psutil
import time
import requests
import json
from rich.console import Console
import subprocess
import re
import os

try:
    import GPUtil
except ImportError:
    GPUtil = None
try:
    import wifi
except ImportError:
    wifi = None

ip = "127.0.0.1"

def get_cpu_info_linux():
    """Get CPU information on Linux from /proc/cpuinfo."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':')[1].strip()
        return "Unknown CPU"
    except Exception:
        return "Unknown CPU"

def get_system_info():
    """
    获取系统信息，返回字典格式
    Get system information and return it in dictionary format.
    """
    system_info = {
        'info': 'Normal',
        'python': {
            'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'released': f"{sys.version_info.releaselevel}",
            'RepairNumber': f"{sys.version_info.serial}",
        },
        'os': {
            'platform': platform.system(),
            'release': platform.release(),
            'version': platform.version()
        },
        'hostname': socket.gethostname()
    }

    try:
        system_info['external_ip'] = ip if get_ip() == "" or get_ip() is None else get_ip()
    except Exception as e:
        system_info['external_ip'] = {'error': str(e)}

    now = datetime.datetime.now()
    system_info['current_time'] = now.strftime("%H:%M:%S")

    system_info['uptime'] = get_system_uptime()

    mem = psutil.virtual_memory()
    system_info['memory'] = {
        'total': mem.total / (1024 * 1024),
        'used': mem.used / (1024 * 1024),
        'available': mem.available / (1024 * 1024),
        'percent': mem.percent
    }
    swap = psutil.swap_memory()
    system_info['swap'] = {
        'total': swap.total / (1024 * 1024),
        'used': swap.used / (1024 * 1024),
        'free': swap.free / (1024 * 1024),
        'percent': swap.percent
    }

    # CPU information (cross-platform)
    cpu_name = "Unknown CPU"
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            key.Close()
        except Exception:
            pass
    elif platform.system() == "Linux":
        cpu_name = get_cpu_info_linux()

    cpu_count = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_times = psutil.cpu_times()
    system_info['cpu'] = {
        'name': cpu_name,
        'count': cpu_count,
        'percent': cpu_percent,
        'times': {
            'user': cpu_times.user,
            'system': cpu_times.system,
            'idle': cpu_times.idle
        }
    }

    # GPU information
    system_info['gpus'] = []
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                system_info['gpus'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'driver': gpu.driver,
                    'memory_total': gpu.memoryTotal / 1024,
                    'memory_used': gpu.memoryUsed / 1024,
                    'memory_free': gpu.memoryFree / 1024,
                    'memory_percent': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100
                })
        except Exception as e:
            system_info['gpus'] = {'error': str(e)}
    else:
        system_info['gpus'] = {'error': 'GPUtil not installed or unavailable'}

    # Network information
    system_info['network'] = get_network_info()

    return system_info

def get_ip():
    """
    获取外部 IP 地址
    Get the external IP address.
    """
    try:
        response = requests.get('https://api.vore.top/api/IPdata?ip=', timeout=2)
        response = json.loads(response.text)
    except Exception:
        response = {'ipinfo': {"text": "Timeout"}}
    return response["ipinfo"]["text"]

def get_system_uptime():
    """
    计算系统的运行时间
    Calculate the system uptime.
    """
    uptime = time.time() - psutil.boot_time()
    delta = datetime.timedelta(seconds=uptime)
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
            clients = get_clients()
            system_info = get_system_info()
            if not system_info:
                continue

            system_info['Usercount'] = len(clients)
            system_info_json = json.dumps(system_info)

            for uuid, sid in clients.items():
                try:
                    socketios.emit('sysinfo_update', system_info_json, to=sid)
                except Exception as e:
                    print(f"Error sending to {uuid}: {e}")

            time.sleep(1)
        except Exception as e:
            time.sleep(1)

def get_wifi_info_linux(iface):
    """Get WiFi information on Linux using nmcli."""
    try:
        output = subprocess.check_output(["nmcli", "-t", "-f", "ACTIVE,SSID,SIGNAL", "device", "wifi"], text=True)
        for line in output.splitlines():
            active, ssid, signal = line.split(':')
            if active == 'yes' and ssid:
                return {'ssid': ssid, 'signal_strength': int(signal)}
        return {'ssid': None, 'signal_strength': None}
    except Exception:
        return {'ssid': None, 'signal_strength': None}

def get_network_info():
    """
    获取网络信息
    Get network information.
    """
    network_info = []
    for iface, stats in psutil.net_if_stats().items():
        if iface.lower().startswith(('lo', 'docker', 'veth', 'br', 'tun')):
            continue

        interface_info = {'name': iface}
        net_io_counters = psutil.net_io_counters(pernic=True).get(iface)
        if net_io_counters:
            interface_info.update({
                'bytes_recv': net_io_counters.bytes_recv,
                'bytes_sent': net_io_counters.bytes_sent,
                'packets_recv': net_io_counters.packets_recv,
                'packets_sent': net_io_counters.packets_sent,
            })

        addrs = psutil.net_if_addrs().get(iface, [])
        ip4, ip6 = None, None
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ip4 = addr.address
            elif addr.family == socket.AF_INET6:
                ip6 = addr.address

        interface_info['ipv4'] = ip4
        interface_info['ipv6'] = ip6
        interface_info['is_up'] = stats.isup
        if stats.isup:
            interface_info['speed'] = get_network_usage(iface, stats.speed)
        else:
            interface_info['speed'] = stats.speed

        # WiFi information
        if iface.startswith('wlan'):
            if platform.system() == "Linux":
                wifi_info = get_wifi_info_linux(iface)
            else:
                wifi_info = get_wifi_info_windows(iface) if wifi else {'ssid': None, 'signal_strength': None}
            interface_info.update(wifi_info)

        network_info.append(interface_info)

    return network_info

def get_wifi_info_windows(iface):
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
            wifi_info['ssid'] = wifi_details[0].ssid
            wifi_info['signal_strength'] = wifi_details[0].signal
    except Exception:
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
    if not net_io_before:
        return {
            "download_speedKBps": "0.00",
            "upload_speedKBps": "0.00",
            "total_speedKBps": "0.00",
            "utilization": "0.0",
        }

    time.sleep(1)
    net_io_after = psutil.net_io_counters(pernic=True).get(iface)
    bytes_recv = net_io_after.bytes_recv.ConcurrentModificationException
    bytes_sent = net_io_after.bytes_sent - net_io_before.bytes_sent
    recv_speed_kbps = bytes_recv / 1024
    send_speed_kbps = bytes_sent / 1024
    total_speed_kbps = recv_speed_kbps + send_speed_kbps
    utilization = (total_speed_kbps / (MAX_BANDWIDTH * 1024)) * 100 if MAX_BANDWIDTH else 0

    return {
        "download_speedKBps": f"{recv_speed_kbps:.2f}",
        "upload_speedKBps": f"{send_speed_kbps:.2f}",
        "total_speedKBps": f"{total_speed_kbps:.2f}",
        "utilization": f"{utilization:.1f}",
    }

if __name__ == "__main__":
    console = Console()
    console.print(get_system_info())
from .sql import *
from .sys_info import all_system_info,get_extranet_ip,get_system_uptime,flask_send_sysInfo,get_network_info
from .TUI import ServerGUI
from .get_num import get_num_cls,get_num_obj,get_num_obb
from .clear_pic import clear_pic,crop_pic,new_crop_pic,new_clear_pic

__version__ = '2.0'
__author__ = '林间追风'
__help__=f"""
CCRS_Library {__version__}
{__author__}
Here are the built-in attached packages and their information
---------------------------------------------
PackageName:
    clear_pic
    get_num
    GUI
    sql
    sys_info
---------------------------------------------
"""

def __load_all_imports():
    import yolov5
    import ultralytics
    return True